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
import time
import os
import csv
from pathlib import Path
import torch
import pickle
from random import randint
from utils.loss_utils import l1_loss, ssim, quaternion_loss, d_xyz_gt, d_rot_gt
from gaussian_renderer import render, network_gui
import sys
import json
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.debug_utils import save_tensor_img
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import re
import ipdb
import matplotlib.pyplot as plt
import math
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
_TB_HIST_DISABLED = False


def compute_storage_metrics(frame_output_path, fps=30.0):
    frame_dir = Path(frame_output_path)
    metrics = {
        "checkpoint_size_kb": 0.0,
        "model_artifact_size_kb": 0.0,
        "entropy_bitstream_size_kb": 0.0,
        "real_bitrate_kbps": 0.0,
    }
    if not frame_dir.exists():
        return metrics

    ckpt_files = list(frame_dir.glob("chkpnt*.pth"))
    if ckpt_files:
        metrics["checkpoint_size_kb"] = max(f.stat().st_size for f in ckpt_files) / 1024.0

    point_cloud_root = frame_dir / "point_cloud"
    if point_cloud_root.exists():
        iter_dirs = sorted(
            [p for p in point_cloud_root.glob("iteration_*") if p.is_dir()],
            key=lambda p: int(p.name.split("_")[-1]),
        )
        if iter_dirs:
            latest_iter = iter_dirs[-1]
            model_files = list(latest_iter.rglob("*.ply"))
            metrics["model_artifact_size_kb"] = sum(f.stat().st_size for f in model_files) / 1024.0

            bitstream_files = [f for f in latest_iter.rglob("feature*") if f.is_file()]
            bitstream_bytes = sum(f.stat().st_size for f in bitstream_files)
            metrics["entropy_bitstream_size_kb"] = bitstream_bytes / 1024.0
            if fps > 0:
                metrics["real_bitrate_kbps"] = (bitstream_bytes * 8.0 * fps) / 1000.0

    return metrics

class rdloss(torch.nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, return_type="all"):
        super().__init__()

        self.metric = torch.nn.MSELoss()
        self.lmbda = lmbda
        self.return_type = return_type

    def forward(self, y_hat,y_likelihoods, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in y_likelihoods
        )
        out["mse_loss"] = self.metric(y_hat, target)
        distortion = out["mse_loss"]

        out["loss"] = self.lmbda * distortion + out["bpp_loss"]

        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]

def training_one_frame(dataset, opt, pipe, load_iteration, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    start_time=time.time()
    last_s1_res = []
    last_s2_res = []
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree, q = dataset.q)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)

    # ===== 新增 Anchor 初始化 =====
    gaussians.init_anchor(num_anchor=1024)
    gaussians.assign_anchor_by_xyz()


    gaussians.training_one_frame_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    s1_start_time=time.time()
    # Train the MEM
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()
                     
        # Query the MEM
        gaussians.query_mem()
        
        loss = torch.tensor(0.).cuda()
        
        
        # A simple 
        for batch_iteraion in range(opt.batch_size):
        
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["depth"],render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            ssim_img = ssim(image,gt_image)
            pixel_l1_map = torch.abs(image - gt_image).mean(dim=0)
            error_q = float(max(0.5, min(0.99, opt.mv_error_quantile)))
            hard_threshold = torch.quantile(pixel_l1_map.detach().reshape(-1), error_q)
            high_error_ratio = (pixel_l1_map.detach() > hard_threshold).float().mean()
            photo_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_img)
            gaussians.update_multiview_consistency(
                visibility_filter=visibility_filter,
                radii=radii,
                photometric_error_scalar=photo_loss.detach().item(),
                high_error_ratio=high_error_ratio.item(),
                ema_decay=opt.mv_ema_decay,
            )
            loss += photo_loss
            loss += 1e-5 * gaussians.mem.model.train_entropy(q=dataset.q) 

        loss/=opt.batch_size
        loss.backward()

        if iteration == opt.iterations/2:
            def adjust_learning_rate(optimizer):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.01 
            adjust_learning_rate(gaussians.mem_optimizer)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            s1_res = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if s1_res is not None:
                last_s1_res.append(s1_res)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration=iteration, save_type='all')

            # Tracking Densification Stats
            if iteration > opt.densify_from_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.output_path + "/chkpnt" + str(iteration) + ".pth")

            if (
                bool(getattr(opt, "s1_mv_prune_enable", False))
                and iteration >= int(getattr(opt, "s1_mv_prune_warmup", 800))
                and iteration % max(1, int(getattr(opt, "s1_mv_prune_interval", 200))) == 0
            ):
                n_pruned = gaussians.prune_stage1_by_multiview(opt)
                if n_pruned > 0:
                    print(f"[Stage1][ITER {iteration}] multi-view pruned points: {n_pruned}")

            # Optimizer step
            if iteration <= opt.iterations:
                gaussians.mem_optimizer.step()
                gaussians.mem_optimizer.zero_grad(set_to_none = True)


    s1_end_time=time.time()
    # Dump the MEM
    scene.dump_MEM()
    # Update Gaussians by MEM
    gaussians.update_by_mem()
    if(opt.iterations_s2>0):
    # Prune, Clone and setting up  
        gaussians.training_one_frame_s2_setup(opt)
        print(f"[Stage2] Initialized added Gaussians: {gaussians._added_xyz.shape[0]}")
        gaussians.assign_anchor_by_xyz()
        gaussians.limit_added_points(opt.max_added_ratio * opt.compression_ratio_s2)
        print(f"[Stage2] Added Gaussians after limit: {gaussians._added_xyz.shape[0]}")
        progress_bar = tqdm(range(opt.iterations, opt.iterations + opt.iterations_s2), desc="Training progress of Stage 2")    
        criterion = rdloss(lmbda=0.01)
    # Train the new Gaussians
    for iteration in range(opt.iterations + 1, opt.iterations + opt.iterations_s2 + 1):        
        iter_start.record()
        
        loss = torch.tensor(0.).cuda()
        for batch_iteraion in range(opt.batch_size):
        
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["depth"],render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            pixel_l1_map = torch.abs(image - gt_image).mean(dim=0)
            error_q = float(max(0.5, min(0.99, opt.mv_error_quantile)))
            hard_threshold = torch.quantile(pixel_l1_map.detach().reshape(-1), error_q)
            high_error_ratio = (pixel_l1_map.detach() > hard_threshold).float().mean()

            per_point_error = torch.zeros((gaussians.get_xyz.shape[0],), device="cuda", dtype=torch.float32)
            per_point_error[visibility_filter] = radii[visibility_filter].detach().to(per_point_error.dtype)
            gaussians.update_transient_mutation_state(
                per_point_error,
                spike_factor=opt.mutation_spike_factor,
                lifetime=opt.transient_lifetime,
            )

            photo_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            gaussians.update_multiview_consistency(
                visibility_filter=visibility_filter,
                radii=radii,
                photometric_error_scalar=photo_loss.detach().item(),
                high_error_ratio=high_error_ratio.item(),
                ema_decay=opt.mv_ema_decay,
            )
            loss += photo_loss

            # 读取 Gaussian feature
            f_dc = gaussians._added_features_dc.contiguous()
            f_rest = gaussians._added_features_rest.contiguous()

            features = torch.cat((f_dc, f_rest), dim=1)

            # ===== Anchor residual =====
            anchor_feat = gaussians.anchor_features[gaussians.anchor_ids]

            residual = features - anchor_feat

            # reshape 为 entropy model 需要的格式
            attributes = residual.view(
                residual.shape[0],
                residual.shape[1],
                3,
                1
            ).permute(3,1,2,0)

            # entropy coding residual
            y_hat, y_likelihoods = gaussians.entropy_bottleneck_added(attributes)

            # RD loss
            codec_loss = criterion(y_hat, y_likelihoods, attributes)['loss']

            # regularization for artifact reduction and compactness
            opacity_sparse = gaussians.get_opacity[-gaussians._added_xyz.shape[0]:].mean() if gaussians._added_xyz.shape[0] > 0 else torch.tensor(0.0, device=loss.device)
            scale_reg = gaussians.get_scaling[-gaussians._added_xyz.shape[0]:].mean() if gaussians._added_xyz.shape[0] > 0 else torch.tensor(0.0, device=loss.device)

            group_rigid_reg = gaussians.apply_group_rigid_motion()
            loss += opt.lambda_rd_base * codec_loss
            loss += opt.lambda_opacity_sparse * opacity_sparse
            loss += opt.lambda_scale_reg * scale_reg
            loss += opt.lambda_group_se3 * group_rigid_reg
            
        loss/=opt.batch_size
        loss.backward()
        
        iter_end.record()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if (iteration - opt.iterations) % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations + opt.iterations_s2:
                progress_bar.close()

            # Log and save
            s2_res = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if s2_res is not None:
                last_s2_res.append(s2_res)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration=iteration, save_type='added')
                scene.save(iteration=iteration, save_type='all')
                             
            if (iteration - opt.iterations) % opt.densification_interval == 0:
                gaussians.adding_and_prune(
                    opt,
                    scene.cameras_extent,
                    force_add=((iteration - opt.iterations) <= opt.densification_interval),
                )

                # ===== 新增：重新分配 anchor =====
                gaussians.assign_anchor_by_xyz()
                gaussians.prune_transient_points()

            # Optimizer step
            if iteration <= opt.iterations + opt.iterations_s2:
                gaussians.optimizer.step()
                stage2_it = iteration - opt.iterations
                if (
                    opt.sh_soft_threshold > 0
                    and stage2_it >= max(0, opt.sh_compress_warmup)
                    and (stage2_it % max(1, opt.sh_compress_interval) == 0)
                ):
                    gaussians.compress_sh_attributes(
                        soft_threshold=opt.sh_soft_threshold,
                        added_only=bool(opt.sh_compress_added_only),
                        opacity_aware=bool(opt.sh_compress_opacity_aware),
                        opacity_alpha=opt.sh_compress_opacity_alpha,
                        preserve_ratio=opt.sh_preserve_ratio,
                        low_opacity_only=bool(opt.sh_compress_low_opacity_only),
                        opacity_cutoff=opt.sh_compress_opacity_cutoff,
                        relative_threshold_cap=opt.sh_threshold_relative_cap,
                        sparsity_ratio=opt.sh_sparsity_ratio,
                        quant_step=opt.sh_quant_step,
                    )
                gaussians.optimizer.zero_grad(set_to_none = True)

    s2_end_time=time.time()
    
    # 计算总训练时间
    pre_time = s1_start_time - start_time
    s1_time = s1_end_time - s1_start_time
    s2_time = s2_end_time - s1_end_time
           
    return last_s1_res, last_s2_res, pre_time, s1_time, s2_time

def prepare_output_and_logger(args):    
    if not args.output_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.output_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.output_path))
    os.makedirs(args.output_path, exist_ok = True)
    with open(os.path.join(args.output_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.output_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    last_test_psnr=0.0
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            #   {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
                              )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    # if scene.gaussians._added_mask is not None:
                    #     added_pkg = renderFunc(viewpoint, scene.gaussians.get_masked_gaussian(scene.gaussians._added_mask), *renderArgs)
                    image, depth = torch.clamp(render_pkg["render"], 0.0, 1.0), render_pkg["depth"]
                    # depth_vis=depth/(depth.max()+1e-5)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_image(config['name'] + "_view_{}/render".format(viewpoint.image_name), image, global_step=iteration)
                        # tb_writer.add_image(config['name'] + "_view_{}/diff".format(viewpoint.image_name), (gt_image-image).abs().mean(dim=0, keepdim=True), global_step=iteration)
                        # tb_writer.add_image(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth_vis, global_step=iteration)
                        # if scene.gaussians._added_mask is not None:
                        #     tb_writer.add_image(config['name'] + "_view_{}/added_gaussians".format(viewpoint.image_name), torch.clamp(added_pkg["render"], 0.0, 1.0), global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_image(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image, global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if config['name'] == 'test':
                    last_test_psnr = psnr_test
                    last_test_image = image
                    last_gt = gt_image
                    last_test_ssim = ssim_test

        global _TB_HIST_DISABLED
        if tb_writer:
            if not _TB_HIST_DISABLED:
                try:
                    tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
                except Exception as e:
                    _TB_HIST_DISABLED = True
                    print(f"[WARN] Disable TensorBoard histogram due to compatibility issue: {e}")
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        
        return {'last_test_psnr':last_test_psnr.cpu().numpy()
                , 'last_test_ssim': last_test_ssim.cpu().numpy()
                , 'last_test_image':last_test_image.cpu()
                , 'last_points_num':scene.gaussians.get_xyz.shape[0]
                # , 'last_gt':last_gt.cpu()
                }

def train_one_frame(lp,op,pp,args):
    args.save_iterations.append(args.iterations + args.iterations_s2)
    if args.depth_smooth==0:
        args.bwd_depth=False
    print("Optimizing " + args.output_path)
    res_dict={}
    if(args.opt_type=='4DGC'):
        s1_ress, s2_ress, pre_time, s1_time, s2_time = training_one_frame(lp.extract(args), op.extract(args), pp.extract(args), args.load_iteration, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

        # All done
        print("\nTraining complete.")
        print(f"Preparation: {pre_time}")
        print(f"Stage 1: {s1_time}")
        print(f"Stage 2: {s2_time}")
        res_dict['preparation_time'] = pre_time
        res_dict['stage1/time'] = s1_time
        res_dict['stage2/time'] = s2_time
        if s1_ress !=[]:
            for idx, s1_res in enumerate(s1_ress):
                save_tensor_img(s1_res['last_test_image'],os.path.join(args.output_path,f'{idx}_rendering1'))
                res_dict[f'stage1/psnr_{idx}']=s1_res['last_test_psnr']
                print(f"Stage1/psnr_{idx}: {s1_res['last_test_psnr']}")
                res_dict[f'stage1/points_num_{idx}']=s1_res['last_points_num']
                res_dict[f'stage1/ssim_{idx}']=s1_res['last_test_ssim']
        if s2_ress !=[]:
            for idx, s2_res in enumerate(s2_ress):
                save_tensor_img(s2_res['last_test_image'],os.path.join(args.output_path,f'{idx}_rendering2'))
                res_dict[f'stage2/psnr_{idx}']=s2_res['last_test_psnr']
                res_dict[f'stage2/points_num_{idx}']=s2_res['last_points_num']
                res_dict[f'stage2/ssim_{idx}']=s2_res['last_test_ssim']
    return res_dict 

def train_frames(lp, op, pp, args):
    # Initialize system state (RNG)
    safe_state(args.quiet)
    video_path=args.video_path
    output_path=args.output_path
    model_path=args.model_path
    load_iteration = args.load_iteration
    sub_paths = os.listdir(video_path)
    pattern = re.compile(r'colmap_(\d+)')
    frames = sorted(
        (item for item in sub_paths if pattern.match(item)),
        key=lambda x: int(pattern.match(x).group(1))
    )
    frames=frames[args.frame_start:args.frame_end]
    if args.frame_start==1:
        args.load_iteration = args.first_load_iteration
    result1_psnr = []
    result2_psnr = []
    result1_ssim = []
    result2_ssim = []
    ckpt_sizes = []
    model_sizes = []
    bitstream_sizes = []
    real_bitrates = []
    csv_rows = []
    for frame in frames:
        start_time = time.time()
        args.source_path = os.path.join(video_path, frame)
        args.output_path = os.path.join(output_path, frame)
        args.model_path = model_path
        
        res_dict = train_one_frame(lp,op,pp,args)

        stage1_psnr = float(res_dict.get('stage1/psnr_0', 0.0))
        stage2_psnr = float(res_dict.get('stage2/psnr_0', stage1_psnr))
        stage1_ssim = float(res_dict.get('stage1/ssim_0', 0.0))
        stage2_ssim = float(res_dict.get('stage2/ssim_0', stage1_ssim))

        result1_psnr.append(stage1_psnr)
        result2_psnr.append(stage2_psnr)
        result1_ssim.append(stage1_ssim)
        result2_ssim.append(stage2_ssim)

        frame_time = time.time()-start_time
        storage_metrics = compute_storage_metrics(args.output_path, fps=float(getattr(args, "fps", 30.0)))
        ckpt_sizes.append(storage_metrics["checkpoint_size_kb"])
        model_sizes.append(storage_metrics["model_artifact_size_kb"])
        bitstream_sizes.append(storage_metrics["entropy_bitstream_size_kb"])
        real_bitrates.append(storage_metrics["real_bitrate_kbps"])
        csv_rows.append({
            'frame': frame,
            'preparation_time': float(res_dict.get('preparation_time', 0.0)),
            'stage1_time': float(res_dict.get('stage1/time', 0.0)),
            'stage2_time': float(res_dict.get('stage2/time', 0.0)),
            'stage1_psnr_0': stage1_psnr,
            'stage1_ssim_0': stage1_ssim,
            'stage1_points_num_0': int(res_dict.get('stage1/points_num_0', 0)),
            'stage2_psnr_0': stage2_psnr,
            'stage2_ssim_0': stage2_ssim,
            'stage2_points_num_0': int(res_dict.get('stage2/points_num_0', 0)),
            'avg_stage1_psnr': float(sum(result1_psnr)/len(result1_psnr)),
            'avg_stage1_ssim': float(sum(result1_ssim)/len(result1_ssim)),
            'avg_stage2_psnr': float(sum(result2_psnr)/len(result2_psnr)),
            'avg_stage2_ssim': float(sum(result2_ssim)/len(result2_ssim)),
            'checkpoint_size_kb': float(storage_metrics['checkpoint_size_kb']),
            'model_artifact_size_kb': float(storage_metrics['model_artifact_size_kb']),
            'entropy_bitstream_size_kb': float(storage_metrics['entropy_bitstream_size_kb']),
            'real_bitrate_kbps': float(storage_metrics['real_bitrate_kbps']),
            'avg_checkpoint_size_kb': float(sum(ckpt_sizes)/len(ckpt_sizes)),
            'avg_model_artifact_size_kb': float(sum(model_sizes)/len(model_sizes)),
            'avg_entropy_bitstream_size_kb': float(sum(bitstream_sizes)/len(bitstream_sizes)),
            'avg_real_bitrate_kbps': float(sum(real_bitrates)/len(real_bitrates)),
            'frame_total_time': float(frame_time),
        })

        output_str = "avg: stage{} PSNR {} SSIM {}".format(1,sum(result1_psnr)/len(result1_psnr),sum(result1_ssim)/len(result1_ssim))
        print(output_str)
        output_str = "avg: stage{} PSNR {} SSIM {}".format(2,sum(result2_psnr)/len(result2_psnr),sum(result2_ssim)/len(result2_ssim))
        print(output_str)

        print(f"Frame {frame} finished in {frame_time} seconds.")
        model_path = args.output_path
        args.load_iteration = load_iteration
        torch.cuda.empty_cache()

    csv_path = os.path.join(output_path, "frame_metrics.csv")
    fieldnames = [
        'frame',
        'preparation_time',
        'stage1_time',
        'stage2_time',
        'stage1_psnr_0',
        'stage1_ssim_0',
        'stage1_points_num_0',
        'stage2_psnr_0',
        'stage2_ssim_0',
        'stage2_points_num_0',
        'avg_stage1_psnr',
        'avg_stage1_ssim',
        'avg_stage2_psnr',
        'avg_stage2_ssim',
        'checkpoint_size_kb',
        'model_artifact_size_kb',
        'entropy_bitstream_size_kb',
        'real_bitrate_kbps',
        'avg_checkpoint_size_kb',
        'avg_model_artifact_size_kb',
        'avg_entropy_bitstream_size_kb',
        'avg_real_bitrate_kbps',
        'frame_total_time',
    ]
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved frame metrics csv to: {csv_path}")


        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--frame_start', type=int, default=1)
    parser.add_argument('--frame_end', type=int, default=150)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--load_iteration', type=int, default=None)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 50, 100])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 50, 100])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--read_config", action='store_true', default=False)
    parser.add_argument("--config_path", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    if args.output_path == "":
        args.output_path=args.model_path
    if args.read_config and args.config_path is not None:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        for key, value in config.items():
            if key not in ["output_path", "source_path", "model_path", "video_path", "debug_from"]:
                setattr(args, key, value)
    serializable_namespace = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, list, dict, tuple, type(None)))}
    json_namespace = json.dumps(serializable_namespace)
    os.makedirs(args.output_path, exist_ok = True)
    with open(os.path.join(args.output_path, "cfg_args.json"), 'w') as f:
        f.write(json_namespace)
    train_frames(lp,op,pp,args)
