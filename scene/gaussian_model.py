#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, strip_symmetric, build_scaling_rotation, build_rotation, quaternion_multiply
from utils.debug_utils import save_cal_graph, save_tensor_img
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, rotate_sh_by_matrix, rotate_sh_by_quaternion
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from mem import Motion_Estimation_Module
import commentjson as ctjs
from scene.Motion_Grid import Motion_Grid
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from scene.entropy_models import EntropyBottleneck

class GaussianModel:

    def setup_functions(self):
        
        # @torch.compile
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
                
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.rotation_compose = quaternion_multiply
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, q : int):
        self.active_sh_degree = 0
        self.q = q
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        
        self._xyz_bound_min = None
        self._xyz_bound_max = None
        
        self._d_xyz = None
        self._d_rot = None
        
        self._new_xyz = None
        self._new_rot = None
        
        self._added_xyz = None
        self._added_features_dc = None
        self._added_features_rest = None
        self._added_opacity = None
        self._added_scaling = None
        self._added_rotation = None
        self._added_mask = None
        self._transient_remaining_life = None
        self._transient_event_score = None
        self._transient_candidate_mask = None
        self._ema_region_error = None
        self._prev_added_xyz = None
        self.mv_add_score = torch.empty(0)
        self.mv_prune_score = torch.empty(0)
        self.mv_seen_views = torch.empty(0)
        
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.color_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def _ensure_multiview_buffers(self):
        n_points = self.get_xyz.shape[0]
        if self.mv_add_score.numel() != n_points:
            self.mv_add_score = torch.zeros((n_points,), device="cuda")
            self.mv_prune_score = torch.zeros((n_points,), device="cuda")
            self.mv_seen_views = torch.zeros((n_points,), device="cuda")

    def reset_multiview_consistency(self):
        self._ensure_multiview_buffers()
        self.mv_add_score.zero_()
        self.mv_prune_score.zero_()
        self.mv_seen_views.zero_()

    def update_multiview_consistency(self, visibility_filter, radii, photometric_error_scalar,
                                     high_error_ratio, ema_decay=0.9):
        self._ensure_multiview_buffers()
        vis_mask = visibility_filter.detach()
        if vis_mask.numel() == 0 or not torch.any(vis_mask):
            return

        visible_radii = radii[vis_mask].detach().to(dtype=torch.float32)
        if visible_radii.numel() == 0:
            return
        radius_norm = visible_radii / (visible_radii.mean() + 1e-6)
        radius_norm = torch.clamp(radius_norm, min=0.25, max=4.0)
        photo = float(photometric_error_scalar)
        hard_ratio = float(high_error_ratio)

        add_signal = torch.clamp(radius_norm * hard_ratio, min=0.0, max=1.0)
        prune_signal = torch.clamp(radius_norm * photo, min=0.0, max=1.0)

        self.mv_add_score[vis_mask] = (
            ema_decay * self.mv_add_score[vis_mask] + (1.0 - ema_decay) * add_signal
        )
        self.mv_prune_score[vis_mask] = (
            ema_decay * self.mv_prune_score[vis_mask] + (1.0 - ema_decay) * prune_signal
        )
        self.mv_seen_views[vis_mask] += 1.0

    def prune_points_stage1(self, prune_mask):
        valid_points_mask = ~prune_mask
        if valid_points_mask.sum() <= 0:
            return
        self._xyz = nn.Parameter(self._xyz[valid_points_mask].detach().requires_grad_(True))
        self._features_dc = nn.Parameter(self._features_dc[valid_points_mask].detach().requires_grad_(True))
        self._features_rest = nn.Parameter(self._features_rest[valid_points_mask].detach().requires_grad_(True))
        self._opacity = nn.Parameter(self._opacity[valid_points_mask].detach().requires_grad_(True))
        self._scaling = nn.Parameter(self._scaling[valid_points_mask].detach().requires_grad_(True))
        self._rotation = nn.Parameter(self._rotation[valid_points_mask].detach().requires_grad_(True))

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.color_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.reset_multiview_consistency()

    def prune_stage1_by_multiview(self, training_args):
        self._ensure_multiview_buffers()
        if self.get_xyz.shape[0] <= 1024:
            return 0
        reliable = self.mv_seen_views >= float(getattr(training_args, "mv_min_views", 6))
        low_mv = self.mv_prune_score < float(getattr(training_args, "s1_mv_prune_threshold", 0.015))
        low_opacity = self.get_opacity.squeeze() < float(getattr(training_args, "s1_mv_opacity_threshold", 0.03))
        candidate_mask = reliable & low_mv & low_opacity
        candidate_idx = torch.where(candidate_mask)[0]
        if candidate_idx.numel() == 0:
            return 0

        max_ratio = float(getattr(training_args, "s1_mv_max_prune_ratio", 0.03))
        max_prune = max(1, int(self.get_xyz.shape[0] * max_ratio))
        prune_k = min(int(candidate_idx.numel()), max_prune)
        scores = self.mv_prune_score[candidate_idx]
        prune_local = torch.topk(scores, k=prune_k, largest=False).indices
        prune_idx = candidate_idx[prune_local]
        prune_mask = torch.zeros((self.get_xyz.shape[0],), device="cuda", dtype=torch.bool)
        prune_mask[prune_idx] = True
        self.prune_points_stage1(prune_mask)
        return int(prune_k)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self._added_scaling is not None:
            return self.scaling_activation(torch.cat((self._scaling, self._added_scaling), dim=0))
        else:
            return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        if self._new_rot is not None:
            return self.rotation_activation(self._new_rot)
        elif self._added_rotation is not None:
            return self.rotation_activation(torch.cat((self._rotation, self._added_rotation), dim=0))
        else:
            return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        if self._new_xyz is not None:
            return self._new_xyz
        elif self._added_xyz is not None:
            return torch.cat((self._xyz, self._added_xyz), dim=0)
        else:
            return self._xyz
    
    @property
    def get_features(self):
        if self._added_features_dc is not None and self._added_features_rest is not None:
            features_dc = torch.cat((self._features_dc, self._added_features_dc), dim=0)
            features_rest = torch.cat((self._features_rest, self._added_features_rest), dim=0)
            return torch.cat((features_dc, features_rest), dim=1)
        else:
            features_dc = self._features_dc
            features_rest = self._features_rest
            return torch.cat((features_dc, features_rest), dim=1)  
          
    @property
    def get_opacity(self):
        if self._added_opacity is not None:
            return self.opacity_activation(torch.cat((self._opacity, self._added_opacity), dim=0))
        else:
            return self.opacity_activation(self._opacity)
        
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self,save_type):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        if save_type == 'added':
            for i in range(self._added_features_rest.shape[1]*self._added_features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
        else:
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def construct_list_of_attributes_ex_sh(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, save_type='all'):
        mkdir_p(os.path.dirname(path))
        if save_type=='added':
            xyz = self._added_xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._added_features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._added_features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._added_opacity.detach().cpu().numpy()
            scale = self._added_scaling.detach().cpu().numpy()
            rotation = self._added_rotation.detach().cpu().numpy()       
        elif save_type=='origin':  
            xyz = self._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()
        elif save_type=='all':
            xyz = self.get_xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self.get_features[:,0:1,:].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self.get_features[:,1:,:].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self.inverse_opacity_activation(self.get_opacity).detach().cpu().numpy()
            scale = self.scaling_inverse_activation(self.get_scaling).detach().cpu().numpy()
            rotation = self.get_rotation.detach().cpu().numpy()
   
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(save_type = save_type)]  
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        if save_type == 'added' and xyz.shape[0]>=4: 
            dtype_ex_sh = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_ex_sh()]
            elements = np.empty(xyz.shape[0], dtype=dtype_ex_sh)
            attributes = np.concatenate((xyz, normals, opacities, scale, rotation), axis=1)
            f_dc = self._added_features_dc.detach()
            f_rest = self._added_features_rest.detach()
            f = torch.cat((f_dc, f_rest), dim=1).view((attributes.shape[0],(self.max_sh_degree + 1) ** 2,3,1)).permute(3,1,2,0)*self.q
            elements[:] = list(map(tuple,attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(path.replace('point_cloud.ply', 'point_cloud_exp_sh.ply'))
            self.entropy_bottleneck_added.compress_range(f[:,:,:,:attributes.shape[0]//4], path=path.replace('point_cloud.ply', 'feature0'))
            self.entropy_bottleneck_added.compress_range(f[:,:,:,(attributes.shape[0]//4):2*(attributes.shape[0]//4)], path=path.replace('point_cloud.ply', 'feature1'))
            self.entropy_bottleneck_added.compress_range(f[:,:,:,(attributes.shape[0]//4)*2:3*(attributes.shape[0]//4)], path=path.replace('point_cloud.ply', 'feature2'))
            self.entropy_bottleneck_added.compress_range(f[:,:,:,(attributes.shape[0]//4)*3:], path=path.replace('point_cloud.ply', 'feature3'))



    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, spatial_lr_scale=0):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.spatial_lr_scale = spatial_lr_scale
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def load_added_ply(self, path, spatial_lr_scale=0):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._added_xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._added_features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._added_features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._added_opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._added_scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._added_rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.spatial_lr_scale = spatial_lr_scale
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def load_added_ply_decompress(self, path, spatial_lr_scale=0):
        if not os.path.exists(path.replace('point_cloud.ply', 'point_cloud_exp_sh.ply')).exists():
            print(f"no compressed file, maybe no added gaussian")
            return 
        plydata = PlyData.read(path.replace('point_cloud.ply', 'point_cloud_exp_sh.ply'))
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f0 = np.array(self.entropy_bottleneck_added.decompress_range(path = path.replace('point_cloud.ply','feature0'))).reshape((self.max_sh_degree + 1) ** 2,3,-1).transpose(2,0,1)
        f1 = np.array(self.entropy_bottleneck_added.decompress_range(path = path.replace('point_cloud.ply','feature1'))).reshape((self.max_sh_degree + 1) ** 2,3,-1).transpose(2,0,1)
        f2 = np.array(self.entropy_bottleneck_added.decompress_range(path = path.replace('point_cloud.ply','feature2'))).reshape((self.max_sh_degree + 1) ** 2,3,-1).transpose(2,0,1)
        f3 = np.array(self.entropy_bottleneck_added.decompress_range(path = path.replace('point_cloud.ply','feature3'))).reshape((self.max_sh_degree + 1) ** 2,3,-1).transpose(2,0,1)

        f = np.concatenate((f0, f1, f2, f3), axis=0)
        features_dc = f[:,:1,:]/self.q
        features_extra = f[:,1:,:]/self.q 
        

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._added_xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._added_features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._added_features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._added_opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._added_scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._added_rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.spatial_lr_scale = spatial_lr_scale
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "entropy_model":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                # print(f"mask shape: {mask.shape}")
                # print(f"stored_state['exp_avg'] shape: {stored_state['exp_avg'].shape}")
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                print(group["name"])
                print(f"mask shape: {mask.shape}, group['params'][0] shape: {group['params'][0].shape}")
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.color_gradient_accum = self.color_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.reset_multiview_consistency()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] == 'entropy_model':
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.color_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.reset_multiview_consistency()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        prune_mask=(self.denom==0).squeeze()
        self.prune_points(prune_mask)
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def adding_postfix(self, added_xyz, added_features_dc, added_features_rest, added_opacities, added_scaling, added_rotation):
        d = {"added_xyz": added_xyz,
        "added_f_dc": added_features_dc,
        "added_f_rest": added_features_rest,
        "added_opacity": added_opacities,
        "added_scaling" : added_scaling,
        "added_rotation" : added_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._added_xyz = optimizable_tensors["added_xyz"]
        self._added_features_dc = optimizable_tensors["added_f_dc"]
        self._added_features_rest = optimizable_tensors["added_f_rest"]
        self._added_opacity = optimizable_tensors["added_opacity"]
        self._added_scaling = optimizable_tensors["added_scaling"]
        self._added_rotation = optimizable_tensors["added_rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.color_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        added_mask=torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.bool)
        added_mask[-self._added_xyz.shape[0]:]=True
        self._added_mask=added_mask
        self._prev_added_xyz = None
        self._reset_transient_tracking()
        self.reset_multiview_consistency()

    def _sync_added_tracking_by_mask(self, valid_points_mask):
        if self._transient_remaining_life is not None and self._transient_remaining_life.numel() > 0:
            self._transient_remaining_life = self._transient_remaining_life[valid_points_mask]
            self._transient_event_score = self._transient_event_score[valid_points_mask]
            self._transient_candidate_mask = self._transient_candidate_mask[valid_points_mask]

        if self.anchor_ids is not None and self.anchor_ids.numel() == valid_points_mask.numel():
            self.anchor_ids = self.anchor_ids[valid_points_mask]

        if self._prev_added_xyz is not None and self._prev_added_xyz.shape[0] == valid_points_mask.numel():
            self._prev_added_xyz = self._prev_added_xyz[valid_points_mask]
        
    def adding_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.adding_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def adding_and_split(self, grads, grad_threshold, std_scale, num_of_split=1, max_added_scale=-1):
        # Extract points that satisfy the gradient condition
        contracted_xyz=self.get_contracted_xyz()                          
        mask = (contracted_xyz >= 0) & (contracted_xyz <= 1)
        mask = mask.all(dim=1)
        num_of_split=num_of_split
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, mask)
        stds = std_scale*self.get_scaling[selected_pts_mask].repeat(num_of_split,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.get_rotation[selected_pts_mask]).repeat(num_of_split,1,1)
        
        added_xyz = (torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(num_of_split, 1)).detach().requires_grad_(True)
        added_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(num_of_split,1) / (0.8*num_of_split)
        )
        if max_added_scale > 0:
            added_scaling = torch.clamp(added_scaling, max=np.log(max_added_scale))
        added_scaling = added_scaling.detach().requires_grad_(True)
        added_rotation = (self.get_rotation[selected_pts_mask].repeat(num_of_split,1)).detach().requires_grad_(True)
        added_features_dc = (self.get_features[:,0:1,:][selected_pts_mask].repeat(num_of_split,1,1)).detach().requires_grad_(True)
        added_features_rest = (self.get_features[:,1:,:][selected_pts_mask].repeat(num_of_split,1,1)).detach().requires_grad_(True)
        added_opacity = (self.inverse_opacity_activation(self.get_opacity[selected_pts_mask]).repeat(num_of_split,1)).detach().requires_grad_(True)

        self.adding_postfix(added_xyz, added_features_dc, added_features_rest, added_opacity, added_scaling, added_rotation)

    def adding_and_prune(self, training_args, extent, force_add=False):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self._ensure_multiview_buffers()
        mv_enabled = bool(getattr(training_args, "mv_consistency_enable", False))
        mv_add_threshold = float(getattr(training_args, "mv_add_threshold", 0.0))
        mv_min_views = float(getattr(training_args, "mv_min_views", 0.0))
        if mv_enabled and mv_add_threshold > 0:
            reliable = self.mv_seen_views >= mv_min_views
            low_consistency = torch.logical_and(reliable, self.mv_add_score < mv_add_threshold)
            grads[low_consistency] = 0.0
        should_add = bool(training_args.s2_adding or force_add)
        if should_add:
            add_before = self._added_xyz.shape[0] if self._added_xyz is not None else 0
            self.adding_and_split(
                grads,
                training_args.densify_grad_threshold,
                training_args.std_scale,
                training_args.num_of_split,
                training_args.max_added_scale,
            )
            add_after = self._added_xyz.shape[0] if self._added_xyz is not None else 0
            min_candidates = max(0, int(getattr(training_args, "s2_min_add_candidates", 0)))
            if add_after <= add_before and min_candidates > 0:
                grad_norm = torch.norm(grads, dim=-1)
                contracted_xyz = self.get_contracted_xyz()
                valid_mask = ((contracted_xyz >= 0) & (contracted_xyz <= 1)).all(dim=1)
                valid_idx = torch.where(valid_mask)[0]
                if valid_idx.numel() > 0:
                    k = min(min_candidates, int(valid_idx.numel()))
                    top_local_idx = torch.topk(grad_norm[valid_idx], k=k, largest=True).indices
                    force_mask = torch.zeros_like(valid_mask)
                    force_mask[valid_idx[top_local_idx]] = True
                    force_grads = torch.zeros_like(grads)
                    force_grads[force_mask] = training_args.densify_grad_threshold + 1.0
                    self.adding_and_split(
                        force_grads,
                        training_args.densify_grad_threshold,
                        training_args.std_scale,
                        training_args.num_of_split,
                        training_args.max_added_scale,
                    )
        if self._added_xyz.shape[0]>0:
            self.prune_added_points(training_args.min_opacity, extent, training_args.max_added_scale, training_args)
            self.limit_added_points(training_args.max_added_ratio)
        torch.cuda.empty_cache()

    def prune_added_points(self, min_opacity, extent, max_added_scale=-1, training_args=None):
        self._ensure_multiview_buffers()
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(prune_mask, big_points_ws)[-self._added_xyz.shape[0]:]
        if max_added_scale > 0:
            over_scale_mask = self.get_scaling[-self._added_xyz.shape[0]:].max(dim=1).values > max_added_scale
            prune_mask = torch.logical_or(prune_mask, over_scale_mask)
        if training_args is not None and bool(getattr(training_args, "mv_consistency_enable", False)):
            mv_prune_threshold = float(getattr(training_args, "mv_prune_threshold", 0.0))
            mv_min_views = float(getattr(training_args, "mv_min_views", 0.0))
            if mv_prune_threshold > 0:
                added_seen = self.mv_seen_views[-self._added_xyz.shape[0]:]
                added_prune_score = self.mv_prune_score[-self._added_xyz.shape[0]:]
                mv_prune_mask = torch.logical_and(added_seen >= mv_min_views, added_prune_score < mv_prune_threshold)
                prune_mask = torch.logical_or(prune_mask, mv_prune_mask)
        valid_points_mask = ~prune_mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._added_xyz = optimizable_tensors["added_xyz"]
        self._added_features_dc = optimizable_tensors["added_f_dc"]
        self._added_features_rest = optimizable_tensors["added_f_rest"]
        self._added_opacity = optimizable_tensors["added_opacity"]
        self._added_scaling = optimizable_tensors["added_scaling"]
        self._added_rotation = optimizable_tensors["added_rotation"]
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.color_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        added_mask=torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.bool)
        added_mask[-self._added_xyz.shape[0]:]=True
        self._added_mask=added_mask
        self._sync_added_tracking_by_mask(valid_points_mask)
        self.reset_multiview_consistency()
        torch.cuda.empty_cache()
        
    def limit_added_points(self, max_added_ratio):
        if self._added_xyz is None or self._added_xyz.shape[0] == 0:
            return
        max_added = int(self._xyz.shape[0] * max_added_ratio)
        if max_added <= 0 or self._added_xyz.shape[0] <= max_added:
            return

        added_opacity = self.get_opacity[-self._added_xyz.shape[0]:].squeeze()
        keep_idx = torch.topk(added_opacity, k=max_added, largest=True).indices
        valid_points_mask = torch.zeros_like(added_opacity, dtype=torch.bool)
        valid_points_mask[keep_idx] = True

        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._added_xyz = optimizable_tensors["added_xyz"]
        self._added_features_dc = optimizable_tensors["added_f_dc"]
        self._added_features_rest = optimizable_tensors["added_f_rest"]
        self._added_opacity = optimizable_tensors["added_opacity"]
        self._added_scaling = optimizable_tensors["added_scaling"]
        self._added_rotation = optimizable_tensors["added_rotation"]

        added_mask=torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.bool)
        added_mask[-self._added_xyz.shape[0]:]=True
        self._added_mask=added_mask
        self._sync_added_tracking_by_mask(valid_points_mask)
        self.reset_multiview_consistency()

    def training_one_frame_s2_setup(self, training_args):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grad_norm = torch.norm(grads, dim=-1)

        contracted_xyz=self.get_contracted_xyz()                          
        mask = (contracted_xyz >= 0) & (contracted_xyz <= 1)
        mask = mask.all(dim=1)

        # Spawn
        num_of_spawn=training_args.num_of_spawn
        selected_pts_mask_spawn = torch.where(torch.norm(grads, dim=-1) >= training_args.densify_grad_threshold, True, False)
        selected_pts_mask_spawn = torch.logical_and(selected_pts_mask_spawn, mask)
        N=selected_pts_mask_spawn.sum()
        stds = training_args.std_scale*self.get_scaling[selected_pts_mask_spawn].repeat(num_of_spawn,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.get_rotation[selected_pts_mask_spawn]).repeat(num_of_spawn,1,1)
        added_xyz_spawn = (torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask_spawn].repeat(num_of_spawn, 1)).detach().requires_grad_(True)
        added_rotation_spawn = torch.tensor([1.,0.,0.,0.],device='cuda').repeat(N*num_of_spawn, 1).detach().requires_grad_(True)
        added_opacity_spawn = self.inverse_opacity_activation(torch.tensor([0.1],device='cuda')).repeat(N*num_of_spawn, 1).detach().requires_grad_(True)
        added_scaling_spawn = (self.scaling_inverse_activation(self.get_scaling[selected_pts_mask_spawn].repeat(num_of_spawn,1) / (0.8*num_of_spawn))).detach().requires_grad_(True)
        if training_args.max_added_scale > 0:
            added_scaling_spawn = torch.clamp(added_scaling_spawn, max=np.log(training_args.max_added_scale)).detach().requires_grad_(True)
        added_features_dc_spawn = (self.get_features[:,0:1,:][selected_pts_mask_spawn].repeat(num_of_spawn,1,1)).detach().requires_grad_(True)
        added_features_rest_spawn = (self.get_features[:,1:,:][selected_pts_mask_spawn].repeat(num_of_spawn,1,1)).detach().requires_grad_(True)

        # Clone
        rotation = 2.0 * torch.acos(torch.clamp(self._d_rot[:,0].abs(), min=0.0, max=1.0))
        xyz_mask = torch.norm(self._d_xyz, dim=-1) >= training_args.xyz_threshold
        rot_mask = torch.norm(rotation, dim=-1) >= training_args.rot_threshold
        scale_mask = torch.norm(self._scaling, dim=-1) >= training_args.scale_threshold
        selected_pts_mask_clone = torch.logical_and(xyz_mask, rot_mask)
        selected_pts_mask_clone = torch.logical_and(selected_pts_mask_clone, scale_mask)
        selected_pts_mask_clone = torch.logical_and(selected_pts_mask_clone, mask)

        if selected_pts_mask_spawn.sum() == 0 and selected_pts_mask_clone.sum() == 0:
            candidate_idx = torch.where(mask)[0]
            if candidate_idx.numel() > 0:
                fallback_k = min(int(training_args.num_of_spawn), int(candidate_idx.numel()))
                top_local_idx = torch.topk(grad_norm[candidate_idx], k=fallback_k, largest=True).indices
                selected_pts_mask_clone[candidate_idx[top_local_idx]] = True

        added_xyz_clone = self.get_xyz[selected_pts_mask_clone].clone().detach().requires_grad_(True)
        added_features_dc_clone = self.get_features[:, 0:1, :][selected_pts_mask_clone].clone().detach().requires_grad_(True)
        added_features_rest_clone = self.get_features[:, 1:, :][selected_pts_mask_clone].clone().detach().requires_grad_(True)
        added_opacity_clone = (self._opacity[selected_pts_mask_clone]/10).clone().detach().requires_grad_(True)
        added_scaling_clone = (self._scaling[selected_pts_mask_clone] - 2).clone().detach().requires_grad_(True)
        if training_args.max_added_scale > 0:
            added_scaling_clone = torch.clamp(added_scaling_clone, max=np.log(training_args.max_added_scale)).detach().requires_grad_(True)
        added_rotation_clone = self.get_rotation[selected_pts_mask_clone].clone().detach().requires_grad_(True)

        # Combine spawned and cloned Gaussians
        total_spawn = added_xyz_spawn.shape[0]
        total_clone = added_xyz_clone.shape[0]

        self._added_xyz = torch.zeros((total_spawn + total_clone, 3), device="cuda", requires_grad=True)
        self._added_features_dc = torch.zeros((total_spawn + total_clone, self._features_dc.shape[1], self._features_dc.shape[2]), device="cuda", requires_grad=True)
        self._added_features_rest = torch.zeros((total_spawn + total_clone, self._features_rest.shape[1], self._features_rest.shape[2]), device="cuda", requires_grad=True)
        self._added_opacity = torch.zeros((total_spawn + total_clone, 1), device="cuda", requires_grad=True)
        self._added_scaling = torch.zeros((total_spawn + total_clone, 3), device="cuda", requires_grad=True)
        self._added_rotation = torch.zeros((total_spawn + total_clone, 4), device="cuda", requires_grad=True)

        if total_spawn > 0:
            self._added_xyz[:total_spawn].data.copy_(added_xyz_spawn)
            self._added_features_dc[:total_spawn].data.copy_(added_features_dc_spawn)
            self._added_features_rest[:total_spawn].data.copy_(added_features_rest_spawn)
            self._added_opacity[:total_spawn].data.copy_(added_opacity_spawn)
            self._added_scaling[:total_spawn].data.copy_(added_scaling_spawn)
            self._added_rotation[:total_spawn].data.copy_(added_rotation_spawn)
    
        if total_clone > 0:
            self._added_xyz[total_spawn:].data.copy_(added_xyz_clone)
            self._added_features_dc[total_spawn:].data.copy_(added_features_dc_clone)
            self._added_features_rest[total_spawn:].data.copy_(added_features_rest_clone)
            self._added_opacity[total_spawn:].data.copy_(added_opacity_clone)
            self._added_scaling[total_spawn:].data.copy_(added_scaling_clone)
            self._added_rotation[total_spawn:].data.copy_(added_rotation_clone)

        # Optimizer
        l = [
            {'params': [self._added_xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "added_xyz"},
            {'params': [self._added_features_dc], 'lr': training_args.feature_lr, "name": "added_f_dc"},
            {'params': [self._added_features_rest], 'lr': training_args.feature_lr / 20.0, "name": "added_f_rest"},
            {'params': [self._added_opacity], 'lr': training_args.opacity_lr, "name": "added_opacity"},
            {'params': [self._added_scaling], 'lr': training_args.scaling_lr, "name": "added_scaling"},
            {'params': [self._added_rotation], 'lr': training_args.rotation_lr, "name": "added_rotation"}
        ]

        self.entropy_bottleneck_added = EntropyBottleneck(channels=(1+self.max_sh_degree)**2,entropy_coder='rangecoder').to('cuda')
        for param in self.entropy_bottleneck_added.parameters():
            l.append({'params': [param], 'lr': 1e-3,  "name": "entropy_model"})
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.color_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
               
        added_mask=torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.bool)
        added_mask[-self._added_xyz.shape[0]:]=True
        self._added_mask=added_mask
        self._prev_added_xyz = None
        self._reset_transient_tracking()
        self.reset_multiview_consistency()

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.color_gradient_accum[update_filter] += torch.norm(self._features_dc.grad[update_filter].squeeze(), dim=-1, keepdim=True)

        self.denom[update_filter] += 1

    def _sanitize_mem_displacement(self, d_xyz):
        if d_xyz is None or d_xyz.numel() == 0:
            return d_xyz

        # Keep normal MEM motion untouched and only suppress extreme outliers.
        # This avoids large black splats while preserving PSNR-sensitive motion details.
        motion_norm = torch.norm(d_xyz, dim=1)
        if motion_norm.numel() == 0:
            return d_xyz

        scene_min, scene_max = self.get_xyz_bound(90)
        scene_diag = torch.norm(scene_max - scene_min).detach()

        q999 = torch.quantile(motion_norm.detach(), 0.999)
        outlier_threshold = torch.clamp(q999 * 3.0, min=scene_diag * 0.03, max=scene_diag * 0.45)
        hard_cap = torch.clamp(q999 * 6.0, min=scene_diag * 0.06, max=scene_diag * 0.80)

        outlier_mask = motion_norm > outlier_threshold
        if not torch.any(outlier_mask):
            return d_xyz

        safe_norm = torch.clamp(motion_norm, min=1e-8)
        scale = torch.ones_like(safe_norm)
        scale[outlier_mask] = torch.clamp(hard_cap / safe_norm[outlier_mask], max=1.0)
        return d_xyz * scale.unsqueeze(-1)

    def _compute_sh_preserve_mask(self, opacity, preserve_ratio):
        if opacity is None or opacity.numel() == 0:
            return None
        keep_ratio = float(max(0.0, min(1.0, preserve_ratio)))
        if keep_ratio <= 0.0:
            return None
        n = opacity.shape[0]
        keep_k = int(max(1, round(n * keep_ratio)))
        keep_k = min(keep_k, n)
        keep_idx = torch.topk(opacity.squeeze(-1), k=keep_k, largest=True).indices
        preserve_mask = torch.zeros((n,), device=opacity.device, dtype=torch.bool)
        preserve_mask[keep_idx] = True
        return preserve_mask

    def query_mem(self):
        mask, self._d_xyz, self._d_rot = self.mem(self._xyz)
        self._d_xyz = self._sanitize_mem_displacement(self._d_xyz)
        self._new_xyz = self._d_xyz + self._xyz
        self._new_rot = self.rotation_compose(self._rotation, self._d_rot)

    def update_by_mem(self):
        self._xyz = self._new_xyz.clone().detach()
        self._rotation = self._new_rot.clone().detach()
        self._new_xyz = None
        self._new_rot = None
        
    def compress_sh_attributes(self, soft_threshold=0.0, added_only=True, opacity_aware=True, opacity_alpha=1.5,
                               preserve_ratio=0.0, low_opacity_only=True, opacity_cutoff=0.25,
                               relative_threshold_cap=0.15, sparsity_ratio=0.35, quant_step=0.0005):
        if soft_threshold <= 0:
            return

        def _soft_shrink(x, threshold):
            return torch.sign(x) * torch.relu(torch.abs(x) - threshold)

        def _build_threshold(sh_tensor, opacity_tensor):
            threshold = torch.full(
                (sh_tensor.shape[0], 1, 1),
                float(soft_threshold),
                device=sh_tensor.device,
                dtype=sh_tensor.dtype,
            )
            if opacity_aware and opacity_tensor is not None:
                threshold = threshold * (1.0 + opacity_alpha * (1.0 - opacity_tensor.unsqueeze(1)))
            if low_opacity_only and opacity_tensor is not None:
                threshold = threshold * (opacity_tensor.unsqueeze(1) <= opacity_cutoff).to(sh_tensor.dtype)
            if relative_threshold_cap > 0:
                sh_scale = sh_tensor.abs().mean(dim=(1, 2), keepdim=True)
                threshold = torch.minimum(threshold, sh_scale * relative_threshold_cap)
            return threshold

        def _sparsify_and_quantize(sh_tensor, original_tensor, opacity_tensor, preserve_mask):
            if sparsity_ratio <= 0 and quant_step <= 0:
                return sh_tensor

            n_points, n_coeff, n_ch = sh_tensor.shape
            work_mask = torch.ones((n_points,), device=sh_tensor.device, dtype=torch.bool)
            if low_opacity_only and opacity_tensor is not None:
                work_mask = opacity_tensor.squeeze(-1) <= opacity_cutoff
            if preserve_mask is not None:
                work_mask = torch.logical_and(work_mask, ~preserve_mask)
            if not torch.any(work_mask):
                return sh_tensor

            compressed = sh_tensor.clone()
            compressed_rows = compressed[work_mask]
            flat = compressed_rows.reshape(compressed_rows.shape[0], -1)

            if sparsity_ratio > 0:
                total_dim = flat.shape[1]
                keep_dim = int(max(1, round(total_dim * (1.0 - float(sparsity_ratio)))))
                keep_dim = min(keep_dim, total_dim)
                topk_idx = torch.topk(flat.abs(), k=keep_dim, dim=1, largest=True).indices
                keep_mask = torch.zeros_like(flat, dtype=torch.bool)
                keep_mask.scatter_(1, topk_idx, True)
                flat = torch.where(keep_mask, flat, torch.zeros_like(flat))

            if quant_step > 0:
                q = float(quant_step)
                flat = torch.round(flat / q) * q

            compressed_rows = flat.view(-1, n_coeff, n_ch)
            compressed[work_mask] = compressed_rows
            if preserve_mask is not None:
                compressed[preserve_mask] = original_tensor[preserve_mask]
            return compressed

        with torch.no_grad():
            if self._added_features_rest is not None and self._added_features_rest.numel() > 0:
                added_sh = self._added_features_rest.data
                added_opacity = None
                if self._added_opacity is not None and self._added_opacity.numel() > 0:
                    added_opacity = self.opacity_activation(self._added_opacity.data).clamp(0.0, 1.0)
                threshold = _build_threshold(added_sh, added_opacity)
                shrunk = _soft_shrink(added_sh, threshold)
                preserve_mask = self._compute_sh_preserve_mask(added_opacity, preserve_ratio)
                compressed = _sparsify_and_quantize(shrunk, added_sh, added_opacity, preserve_mask)
                self._added_features_rest.data.copy_(compressed)

            if (not added_only) and self._features_rest is not None and self._features_rest.numel() > 0:
                base_sh = self._features_rest.data
                base_opacity = None
                if self._opacity is not None and self._opacity.numel() > 0:
                    base_opacity = self.opacity_activation(self._opacity.data).clamp(0.0, 1.0)
                threshold = _build_threshold(base_sh, base_opacity)
                shrunk = _soft_shrink(base_sh, threshold)
                preserve_mask = self._compute_sh_preserve_mask(base_opacity, preserve_ratio)
                compressed = _sparsify_and_quantize(shrunk, base_sh, base_opacity, preserve_mask)
                self._features_rest.data.copy_(compressed)

    def get_contracted_xyz(self):
        with torch.no_grad():
            xyz = self.get_xyz
            xyz_bound_min, xyz_bound_max = self.get_xyz_bound(90)
            normalzied_xyz=(xyz-xyz_bound_min)/(xyz_bound_max-xyz_bound_min)
            return normalzied_xyz
    
    def get_xyz_bound(self, percentile=90):
        with torch.no_grad():
            if self._xyz_bound_min is None:
                half_percentile = (100 - percentile) / 200
                self._xyz_bound_min = torch.quantile(self._xyz,half_percentile,dim=0)
                self._xyz_bound_max = torch.quantile(self._xyz,1 - half_percentile,dim=0)
            return self._xyz_bound_min, self._xyz_bound_max       
                     
    def training_one_frame_setup(self,training_args):
        print('training_one_frame_setup')
        model = Motion_Grid(q = self.q).to(torch.device("cuda"))
        self.mem=Motion_Estimation_Module(model,self.get_xyz_bound()[0],self.get_xyz_bound()[1])
        self.mem.load_state_dict(torch.load(training_args.mem_path),strict = False)
        
        self._xyz_bound_min = self.mem.xyz_bound_min
        self._xyz_bound_max = self.mem.xyz_bound_max
        self.mem_optimizer = torch.optim.Adam(self.mem.model.get_optparam_groups())  
        self.scheduler = lr_scheduler.StepLR(self.mem_optimizer, step_size=10, gamma=0.1)
                 
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.color_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.reset_multiview_consistency()
        
    def get_masked_gaussian(self, mask):        
        new_gaussian = GaussianModel(self.max_sh_degree)
        new_gaussian._xyz = self.get_xyz[mask].detach()
        new_gaussian._features_dc = self.get_features[:,0:1,:][mask].detach()
        new_gaussian._features_rest = self.get_features[:,1:,:][mask].detach()
        new_gaussian._scaling = self.scaling_inverse_activation(self.get_scaling)[mask].detach()
        new_gaussian._rotation = self.get_rotation[mask].detach()
        new_gaussian._opacity = self.inverse_opacity_activation(self.get_opacity)[mask].detach()
        new_gaussian.xyz_gradient_accum = torch.zeros((new_gaussian._xyz.shape[0], 1), device="cuda")
        new_gaussian.color_gradient_accum = torch.zeros((new_gaussian._xyz.shape[0], 1), device="cuda")
        new_gaussian.denom = torch.zeros((new_gaussian._xyz.shape[0], 1), device="cuda")
        new_gaussian.max_radii2D = torch.zeros((new_gaussian._xyz.shape[0]), device="cuda")
        return new_gaussian
    
    def query_mem_eval(self):
        with torch.no_grad():
            self.mem.model.is_train = False
            mask, self._d_xyz, self._d_rot = self.mem(self.get_xyz)
            self._d_xyz = self._sanitize_mem_displacement(self._d_xyz)
            self._new_xyz = self._d_xyz + self._xyz
            self._new_rot = self.rotation_compose(self._rotation, self._d_rot)



    def _reset_transient_tracking(self):
        if self._added_xyz is None or self._added_xyz.shape[0] == 0:
            self._transient_remaining_life = torch.empty(0, device="cuda", dtype=torch.int32)
            self._transient_event_score = torch.empty(0, device="cuda")
            self._transient_candidate_mask = torch.empty(0, device="cuda", dtype=torch.bool)
            self._ema_region_error = {}
            return
        n_added = self._added_xyz.shape[0]
        self._transient_remaining_life = torch.zeros((n_added,), device="cuda", dtype=torch.int32)
        self._transient_event_score = torch.zeros((n_added,), device="cuda")
        self._transient_candidate_mask = torch.zeros((n_added,), device="cuda", dtype=torch.bool)
        self._ema_region_error = {}

    def update_transient_mutation_state(self, render_error, spike_factor=2.5, lifetime=12, grid_size=0.2):
        if self._added_xyz is None or self._added_xyz.shape[0] == 0:
            return
        n_added = self._added_xyz.shape[0]
        if self._transient_remaining_life is None or self._transient_remaining_life.numel() != n_added:
            self._reset_transient_tracking()

        xyz_added = self._added_xyz.detach()
        error_added = render_error[-n_added:].detach()
        region = torch.floor(xyz_added / grid_size).long()
        region_hash = (region[:, 0] * 73856093 + region[:, 1] * 19349663 + region[:, 2] * 83492791)

        unique_region = torch.unique(region_hash)
        for rid in unique_region:
            mask = region_hash == rid
            cur_err = error_added[mask].mean()
            key = int(rid.item())
            if key not in self._ema_region_error:
                self._ema_region_error[key] = cur_err
                continue
            prev = self._ema_region_error[key]
            ratio = (cur_err + 1e-6) / (prev + 1e-6)
            if ratio > spike_factor:
                self._transient_remaining_life[mask] = int(lifetime)
                self._transient_event_score[mask] = ratio
                self._transient_candidate_mask[mask] = True
            self._ema_region_error[key] = 0.95 * prev + 0.05 * cur_err

        self._transient_remaining_life = torch.clamp(self._transient_remaining_life - 1, min=0)

    def apply_group_rigid_motion(self):
        if self._added_xyz is None or self._added_xyz.shape[0] == 0:
            return torch.tensor(0.0, device="cuda")

        added_xyz = self._added_xyz
        if self._prev_added_xyz is None or self._prev_added_xyz.shape[0] != added_xyz.shape[0]:
            self._prev_added_xyz = added_xyz.detach().clone()
            return torch.tensor(0.0, device=added_xyz.device)

        if self.anchor_ids is None or self.anchor_ids.numel() != added_xyz.shape[0]:
            self.assign_anchor_by_xyz()

        group_ids = self.anchor_ids
        unique_gid = torch.unique(group_ids)
        reg = torch.tensor(0.0, device=added_xyz.device)
        displacement = added_xyz - self._prev_added_xyz
        valid_group = 0

        for gid in unique_gid:
            mask = group_ids == gid
            if mask.sum() < 3:
                continue
            gid_disp = displacement[mask]
            reg = reg + (gid_disp - gid_disp.mean(dim=0, keepdim=True)).pow(2).mean()
            valid_group += 1

        self._prev_added_xyz = added_xyz.detach().clone()
        return reg / max(valid_group, 1)

    def prune_transient_points(self):
        if self._added_xyz is None or self._added_xyz.shape[0] == 0:
            return
        if self._transient_remaining_life is None or self._transient_remaining_life.numel() == 0:
            return
        if (
            self._transient_remaining_life.numel() != self._added_xyz.shape[0]
            or self._transient_candidate_mask is None
            or self._transient_candidate_mask.numel() != self._added_xyz.shape[0]
            or self._transient_event_score is None
            or self._transient_event_score.numel() != self._added_xyz.shape[0]
        ):
            self._reset_transient_tracking()
            return

        valid_points_mask = torch.logical_or(
            ~self._transient_candidate_mask,
            self._transient_remaining_life > 0,
        )
        if valid_points_mask.all():
            return

        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._added_xyz = optimizable_tensors["added_xyz"]
        self._added_features_dc = optimizable_tensors["added_f_dc"]
        self._added_features_rest = optimizable_tensors["added_f_rest"]
        self._added_opacity = optimizable_tensors["added_opacity"]
        self._added_scaling = optimizable_tensors["added_scaling"]
        self._added_rotation = optimizable_tensors["added_rotation"]
        self._sync_added_tracking_by_mask(valid_points_mask)

        added_mask = torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.bool)
        if self._added_xyz.shape[0] > 0:
            added_mask[-self._added_xyz.shape[0]:] = True
        self._added_mask = added_mask

    def assign_anchor_by_xyz(self, grid_size=0.1):
        """
        根据Gaussian的空间位置分配anchor
        """
        xyz = self._added_xyz.detach() if self._added_xyz is not None else self.get_xyz.detach()
        # 计算空间grid
        grid = torch.floor(xyz / grid_size).long()
        # 将3D grid映射成1D id
        anchor_ids = (
            grid[:, 0] * 73856093 +
            grid[:, 1] * 19349663 +
            grid[:, 2] * 83492791
        )
    # 归一化到 anchor 数量范围
        num_anchor = self.anchor_features.shape[0]
        anchor_ids = torch.abs(anchor_ids) % num_anchor
        self.anchor_ids = anchor_ids
        return anchor_ids

    def init_anchor(self, num_anchor=1024):
        """
        初始化 anchor feature
        """
        feat_dim = self._features_dc.shape[1] + self._features_rest.shape[1]
        feat_channel = self._features_dc.shape[2]
        self.anchor_features = torch.nn.Parameter(
            torch.zeros(num_anchor, feat_dim, feat_channel).cuda()
        )
        # 每个 Gaussian 对应一个 anchor
        target_xyz = self._added_xyz if self._added_xyz is not None else self.get_xyz
        self.anchor_ids = torch.randint(
            0, num_anchor, (target_xyz.shape[0],), device="cuda"
        )


    def compute_anchor_residual(self):

        f_dc = self._added_features_dc
        f_rest = self._added_features_rest
        features = torch.cat((f_dc, f_rest), dim=1)

        if self.anchor_ids.shape[0] != features.shape[0]:
            self.assign_anchor_by_xyz()

        anchor_feat = self.anchor_features[self.anchor_ids]
        residual = features - anchor_feat
        return residual


    def reconstruct_feature(self, residual):
        anchor_feat = self.anchor_features[self.anchor_ids]
        feature = anchor_feat + residual
        return feature


    def minibatch_kmeans_anchor(
            self,
            num_anchor=1024,
            batch_size=8192,
            num_iters=500,
            lr=0.5
        ):
        """
        MiniBatch K-means for Gaussian feature anchors
        """
        f_dc = self._added_features_dc.contiguous()
        f_rest = self._added_features_rest.contiguous()
        features = torch.cat((f_dc, f_rest), dim=1)
        device = features.device
        N, Fdim = features.shape
    
        # 初始化 anchor
        rand_idx = torch.randperm(N)[:num_anchor]
        anchors = features[rand_idx].clone()
        counts = torch.zeros(num_anchor, device=device)

        for i in range(num_iters):

            # 随机采样 batch
            idx = torch.randint(0, N, (batch_size,), device=device)
            batch = features[idx]

            # 距离
            dist = torch.cdist(batch, anchors)

            # 最近 anchor
            nearest = torch.argmin(dist, dim=1)

            for j in range(batch_size):

                k = nearest[j]

                counts[k] += 1

                eta = lr / counts[k]

                anchors[k] = (1 - eta) * anchors[k] + eta * batch[j]

            if i % 50 == 0:
                print(f"KMeans iter {i}/{num_iters}")

        # 最终分配 anchor id
        dist_full = torch.cdist(features, anchors)
        anchor_ids = torch.argmin(dist_full, dim=1)

        self.anchor_features = torch.nn.Parameter(anchors)
        self.anchor_ids = anchor_ids

        return anchor_ids


class GaussianModel_base:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, is_train: bool, q: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self.is_train = is_train
        self.q = q
        self.entropy_bottleneck = EntropyBottleneck(channels=(1+sh_degree)**2,entropy_coder='rangecoder').to('cuda')
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self.is_train:
            noise_scale = (torch.max(self._scaling) - torch.min(self._scaling)) / 255.0
            noise = torch.rand_like(self._scaling) * noise_scale - noise_scale / 2.0
            return self.scaling_activation(self._scaling+noise)
        else:    
            return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        if self.is_train:
            noise_scale = (torch.max(self._rotation) - torch.min(self._rotation)) / 255.0
            noise = torch.rand_like(self._rotation) * noise_scale - noise_scale / 2.0
            return self.rotation_activation(self._rotation + noise)
        else:    
            return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        if self.is_train:
            noise_scale = (torch.max(self._xyz) - torch.min(self._xyz)) / 65535.0
            noise = torch.rand_like(self._xyz) * noise_scale - noise_scale / 2.0
            return self._xyz + noise
        else:    
            return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        feature = torch.cat((features_dc, features_rest), dim=1)
        if self.is_train:
            # return feature
            half = 0.5/self.q
            noise = torch.rand(feature.shape,device = 'cuda')/self.q-half
            return feature+noise
        else:
            return feature
    
    @property
    def get_opacity(self):
        if self.is_train:
            noise_scale = (torch.max(self._opacity) - torch.min(self._opacity)) / 255.0
            noise = torch.rand_like(self._opacity) * noise_scale - noise_scale / 2.0
            return self.opacity_activation(self._opacity+noise)
        else:
            return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        for param in self.entropy_bottleneck.parameters():
            l.append({'params': [param], 'lr': 1e-3,  "name": "entropy_model"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def construct_list_of_attributes_full(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc_full = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest_full = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_dc = self._features_dc.detach()
        f_rest = self._features_rest.detach()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        dtype_ex_sh = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_full()]

        elements = np.empty(xyz.shape[0], dtype=dtype_ex_sh)
        elements_full = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes_full = np.concatenate((xyz, normals, f_dc_full, f_rest_full, opacities, scale, rotation), axis=1)
        attributes = np.concatenate((xyz, normals, opacities, scale, rotation), axis=1)
        f = torch.cat((f_dc, f_rest), dim=1).view((attributes.shape[0],(self.max_sh_degree + 1) ** 2,3,1)).permute(3,1,2,0)*self.q
        elements[:] = list(map(tuple, attributes))
        elements_full[:] = list(map(tuple, attributes_full))
        el_full = PlyElement.describe(elements_full, 'vertex')
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path.replace('point_cloud.ply', 'point_cloud_exp_sh.ply'))
        # PlyData([el_full]).write(path)
        PlyData([el_full]).write(path)

        self.entropy_bottleneck.compress_range(f[:,:,:,:attributes.shape[0]//4], path=path.replace('point_cloud.ply', 'feature0'))
        self.entropy_bottleneck.compress_range(f[:,:,:,(attributes.shape[0]//4):2*(attributes.shape[0]//4)], path=path.replace('point_cloud.ply', 'feature1'))
        self.entropy_bottleneck.compress_range(f[:,:,:,(attributes.shape[0]//4)*2:3*(attributes.shape[0]//4)], path=path.replace('point_cloud.ply', 'feature2'))
        self.entropy_bottleneck.compress_range(f[:,:,:,(attributes.shape[0]//4)*3:], path=path.replace('point_cloud.ply', 'feature3'))

        

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs exceps DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))

        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def load_decompress_ply(self, path):
        plydata = PlyData.read(path.replace('point_cloud.ply', 'point_cloud_exp_sh.ply'))
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f0 = np.array(self.entropy_bottleneck.decompress_range(path = path.replace('point_cloud.ply','feature0'))).reshape((self.max_sh_degree + 1) ** 2,3,-1).transpose(2,0,1)
        f1 = np.array(self.entropy_bottleneck.decompress_range(path = path.replace('point_cloud.ply','feature1'))).reshape((self.max_sh_degree + 1) ** 2,3,-1).transpose(2,0,1)
        f2 = np.array(self.entropy_bottleneck.decompress_range(path = path.replace('point_cloud.ply','feature2'))).reshape((self.max_sh_degree + 1) ** 2,3,-1).transpose(2,0,1)
        f3 = np.array(self.entropy_bottleneck.decompress_range(path = path.replace('point_cloud.ply','feature3'))).reshape((self.max_sh_degree + 1) ** 2,3,-1).transpose(2,0,1)

        f = np.concatenate((f0, f1, f2, f3), axis=0)
        features_dc = f[:,:1,:]/self.q
        features_extra = f[:,1:,:]/self.q 

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "entropy_model":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] == 'entropy_model':
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    
    def init_anchor(self, num_anchor=1024):
        """
        初始化 anchor feature
        """
        feat_dim = self._features_dc.shape[1] + self._features_rest.shape[1]
        feat_channel = self._features_dc.shape[2]
        self.anchor_features = torch.nn.Parameter(
            torch.zeros(num_anchor, feat_dim, feat_channel).cuda()
        )
        # 每个 Gaussian 对应一个 anchor
        self.anchor_ids = torch.randint(
            0, num_anchor, (self.get_xyz.shape[0],), device="cuda"
        )


    def compute_anchor_residual(self):

        f_dc = self._added_features_dc
        f_rest = self._added_features_rest
        features = torch.cat((f_dc, f_rest), dim=1)
        anchor_feat = self.anchor_features[self.anchor_ids]
        residual = features - anchor_feat
        return residual


    def reconstruct_feature(self, residual):
        anchor_feat = self.anchor_features[self.anchor_ids]
        feature = anchor_feat + residual
        return feature
