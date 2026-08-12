#
# Copyright (C) 2025 - 2026, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Options for data loading and training")

    ## Data and Images
    parser.add_argument('-s', '--source_path', type=str, required=True,
                        help="Path to the data folder (should have sparse/0/ if using COLMAP or evaluating poses)")
    parser.add_argument('-i', '--images_dir', type=str, default="images",
                        help="source_path/images_dir is the path to the images (with extensions jpg, png, jpeg, or webp).")
    parser.add_argument('--masks_dir', type=str, default="", 
                        help="If set, source_path/masks_dir is the path to optional masks to apply to the images before computing loss (png).")
    parser.add_argument('--num_loader_threads', type=int, default=4,
                        help="Number of workers to load and prepare input images")
    parser.add_argument('--downsampling', type=float, default=-1.0, help="Downsampling ratio for input images")
    parser.add_argument('--start_at', type=int, default=0,
                        help="Number of frames to skip from the dataset.")
    parser.add_argument('--shuffle', action='store_true',
                        help="Shuffle the input images with a fixed random seed.")

    parser.add_argument('--sh_degree', type=int, default=3)

    ## COLMAP options
    parser.add_argument('--eval_poses', action='store_true',
                        help="Compare poses to COLMAP")
    parser.add_argument('--use_colmap_poses', action='store_true',
                        help="Load COLMAP data for pose and intrinsics initialization")

    ## Learning Rates
    parser.add_argument('--lr_poses', type=float, default=1e-5, help="Pose learning rate")
    parser.add_argument('--position_lr', type=float, default=0.00005, 
                        help="Initial position learning rate")
    parser.add_argument('--feature_lr', type=float, default=0.01, help="Feature learning rate")
    parser.add_argument('--opacity_lr', type=float, default=0.1, help="Opacity learning rate")
    parser.add_argument('--scaling_lr', type=float, default=0.02, help="Scaling learning rate")
    parser.add_argument('--rotation_lr', type=float, default=0.005, help="Rotation learning rate")
    parser.add_argument('--adam_beta1', type=float, default=0.5, help="Adam beta1 (momentum)")
    parser.add_argument('--adam_beta2', type=float, default=0.99, help="Adam beta2 (second moment)")
    parser.add_argument('--lr_colour_corr', type=float, default=1e-3, 
                        help="Exposure compensation learning rate")
    parser.add_argument('--exposure_adam_beta1', type=float, default=0.2, help="Adam beta1 for exposure/colour optimizer")
    parser.add_argument('--exposure_adam_beta2', type=float, default=0.995, help="Adam beta2 for exposure/colour optimizer")
        
    ## Training schedule and losses
    parser.add_argument('--lambda_dssim', type=float, default=0.15, help="Weight for DSSIM loss")
    parser.add_argument('--num_iterations', type=int, default=30,
                        help="Number of training iterations per keyframe")
    parser.add_argument('--save_at_finetune_epoch', type=int, nargs='+', default=[],
                        help="Enable finetuning after the initial on-the-fly reconstruction and save the scene at the end of the specified epochs when fine-tuning.")
    parser.add_argument('--use_last_frame_proba', type=float, default=0.05,
                        help="Probability of using the last registered frame for each training iteration")
    ## Pose initialization options
    # Matching
    parser.add_argument('--min_displacement', type=float, default=2.0e-2,
                        help="Minimum median keypoint displacement for a new keyframe to be added. Relative to the image diagonal.")
    parser.add_argument('--min_init_matches_count', type=int, default=200,
                        help="Minimum number of matched keypoints for a frame to be considered.")
    parser.add_argument('--min_bootstrap_matches_count', type=int, default=400,
                        help="Minimum number of matched keypoints during bootstrap (initial pair selection and frame shelving).")
    parser.add_argument('--num_kpts', type=int, default=4096,
                        help="Number of keypoints to extract from each image")
    parser.add_argument('--match_min_cossim', type=float, default=0.82,
                        help="Minimum cosine similarity for mutual nearest neighbor feature matching")
    parser.add_argument('--match_max_error', type=float, default=1.5e-3,
                        help="Maximum reprojection error for matching keypoints, proportion of the image width. This is used to filter outliers and discard points at triangulation.")
    parser.add_argument('--fundmat_samples', type=int, default=1000,
                        help="Maximum number of set of matches used to estimate the fundamental matrix for outlier removal")
    parser.add_argument('--lg_n_layers', type=int, default=None,
                        help="Number of LightGlue transformer layers (default: 6). Fewer layers = faster matching with some quality tradeoff.")
    parser.add_argument('--min_num_inliers', type=int, default=50,
                        help="The keyframe will be added only if the number of inliers is greater than this value")
    parser.add_argument('--min_3d_points', type=int, default=500,
                        help="Minimum number of 3D points required to add a keyframe")
    parser.add_argument('--min_match_count', type=int, default=10,
                        help="Minimum number of matches required when matching keyframes")
    parser.add_argument('--min_matches_edge', type=int, default=500,
                        help="Minimum number of matches for an edge to be considered robust")
    parser.add_argument('--use_lightglue', action=argparse.BooleanOptionalAction, default=True,
                        help="Use LighterGlue matcher instead of mutual NN")
    parser.add_argument('--use_rotated_descriptors', action=argparse.BooleanOptionalAction, default=True,
                        help="Extract descriptors for all 4 rotations (0°, 90°, 180°, 270°) for rotation-invariant matching")
    # Depth Anything 3 options
    parser.add_argument('--da3_model_dir', type=str, default="depth-anything/DA3-BASE",
                        help="Depth Anything 3 model name or local path.")
    # Initial mini bundle adjustment
    parser.add_argument('--num_keyframes_miniba_bootstrap', type=int, default=8,
                        help="Number of first keyframes accumulated for pose and focal estimation before optimization")
    parser.add_argument('--num_pts_miniba_bootstrap', type=int, default=8*250,
                        help="Number of keypoints considered for initial mini bundle adjustment")
    parser.add_argument('--iters_miniba_bootstrap', type=int, default=75)
    # Bootstrap scale normalization
    parser.add_argument('--scale_depth_target', type=float, default=1.0,
                        help="Target depth value used to normalize the initial scene scale")
    parser.add_argument('--scale_depth_quantile', type=float, default=0.2,
                        help="Depth quantile used for initial scene scale normalization")
    # Focal estimation
    parser.add_argument('--fix_focal', action='store_true',
                        help="If set, will use init_focal or init_fov without reoptimizing focal")
    parser.add_argument('--init_focal', type=float, default=-1.0, 
                        help="Initial focal length in pixels. If not set, will use init_fov or be set as 0.7*width of the image if init_fov is also not set")
    parser.add_argument('--init_fov', type=float, default=-1.0, 
                        help="Initial horizontal FoV in degrees. Used only if init_focal is not set")
    # Incremental pose optimization
    parser.add_argument('--triangulator_max_error', type=float, default=3e-3,
                        help="Maximum reprojection error for triangulation, proportion of the image diagonal")
    parser.add_argument('--triangulator_min_dis', type=float, default=1e-3,
                        help="Minimum disparity threshold for triangulation, proportion of the image diagonal")
    parser.add_argument('--num_prev_keyframes_miniba_incr', type=int, default=5,
                        help="Number of previous keyframes for incremental pose initialization")
    parser.add_argument('--num_prev_keyframes_check', type=int, default=20,
                        help="Number of previous keyframes to check for matches with new keyframe")
    parser.add_argument('--pnp_max_error', type=float, default=5e-3,
                        help="Maximum reprojection error for PnP RANSAC, proportion of the image diagonal.")
    parser.add_argument('--ba_outlier_threshold', type=float, default=6e-3,
                        help="Hard outlier rejection threshold for bundle adjustment, proportion of the image diagonal.")
    parser.add_argument('--pnpransac_samples', type=int, default=2000,
                        help="Number of set of 2D-3D matches used to estimate the initial pose and outlier removal")
    parser.add_argument('--num_pts_miniba_incr', type=int, default=2000,
                        help="Number of keypoints considered for initial mini bundle adjustment")
    parser.add_argument('--iters_miniba_incr', type=int, default=40)
    parser.add_argument('--huber_delta_ratio', type=float, default=1.0,
                        help="Huber delta as a fraction of ba_outlier_threshold for incremental and local BA (0 to disable)")
    parser.add_argument('--baseline_thresh', type=float, default=1e2,
                        help="Minimum scale_depth_quantile to baseline ratio required for sufficient parallax")
    # Bundle adjustment
    parser.add_argument('--deterministic_poses', action=argparse.BooleanOptionalAction, default=False,
                        help="Deterministic pose estimation. Slower and incompatible with lr_poses")
    parser.add_argument('--ba_min_pts_per_keyframe', type=int, default=50,
                        help="Minimum number of BA points per keyframe required to run optimization")
    parser.add_argument('--global_ba_pts_per_keyframe', type=int, default=200,
                        help="Number of points per keyframe to select for global bundle adjustment")
    parser.add_argument('--num_global_ba_rounds', type=int, default=0,
                        help="Number of global BA rounds to run each time a keyframe is added")
    # Local BA
    parser.add_argument('--num_keyframes_localba', type=int, default=20)
    parser.add_argument('--num_pts_localba', type=int, default=20*1000)
    parser.add_argument('--iters_localba', type=int, default=10)
    parser.add_argument('--localba_outlier_mad_scale', type=float, default=-1.0,
                        help="If > 0, local BA also rejects observations with error above median + this*MAD. "
                             "-1 disables.")
    parser.add_argument('--opt_f_until', type=int, default=50)
    parser.add_argument('--localba_fixed_ratio', type=float, default=0.3,
                        help="Fraction of local BA keyframes randomly fixed during each optimization iteration")
    # Loop closure
    parser.add_argument('--min_loop_size', type=int, default=10,
                        help="If the shortest matching path between two keyframes is longer than this, we consider them far apart")
    parser.add_argument('--lc_move_gaussians', action=argparse.BooleanOptionalAction, default=True,
                        help="Move Gaussians after loop closure")

    ## Rendering
    parser.add_argument('--render_near_plane', type=float, default=0.01,
                        help="Near clipping plane distance for rasterizer")
    parser.add_argument('--render_far_plane', type=float, default=10000.0,
                        help="Far clipping plane distance for rasterizer")
    ## Gaussian pruning options
    parser.add_argument('--prune_opacity_threshold', type=float, default=0.05,
                        help="Gaussians with opacity below this threshold are pruned")
    parser.add_argument('--prune_max_screen_size', type=float, default=0.5,
                        help="Gaussians whose screen size exceeds this fraction of image diagonal are pruned")
    parser.add_argument('--coarse_removal_count', type=int, default=20,
                        help="Remove a Gaussian if it is the main contributor for at least this many accurately-sampled new points (i.e. it is coarser than the incoming geometry). Set to 0 to disable coarse-primitive removal.")
    parser.add_argument('--coarse_removal_kf_window', type=int, default=10,
                        help="Protect Gaussians created within this many keyframes of the current one from coarse-primitive removal")

    ## Gaussian initialization options
    parser.add_argument('--depth_conf_threshold', type=float, default=0.25,
                        help="Minimum monocular depth confidence required to use a sampled point for Gaussian initialization")
    parser.add_argument('--min_pts3d_for_depth_align', type=int, default=10,
                        help="Minimum number of 3D-tracked points required to run monocular depth alignment")
    parser.add_argument('--occlusion_depth_factor', type=float, default=1.5,
                        help="Sampled points with depth exceeding this factor times the rendered depth are discarded as occluded")
    parser.add_argument('--init_proba_scaler', type=float, default=2.0,
                        help="Scale the laplacian-based probability of using a pixel to make a new Gaussian primitive. Set to 0 to only use triangulated points.")
    parser.add_argument('--init_opacity_accurate', type=float, default=0.1,
                        help="Initial opacity for depth-sampled Gaussians at accurately tracked points")
    parser.add_argument('--init_opacity_inaccurate', type=float, default=0.02,
                        help="Initial opacity for depth-sampled Gaussians at inaccurately tracked points")
    parser.add_argument('--init_opacity_triangulated', type=float, default=0.3,
                        help="Initial opacity for triangulated Gaussians")
    parser.add_argument('--skip_gs', action=argparse.BooleanOptionalAction, default=False,
                        help="Skip Gaussian splatting reconstruction")    

    ## Hierarchy options
    parser.add_argument('--hierarchy', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--hierarchy_update_freq', type=int, default=10,
                        help="Run hierarchy creation every this many keyframes")
    parser.add_argument('--render_tau', type=float, default=1.0,
                        help="Initial render opacity threshold tau for hierarchy cut (can be adjusted at runtime in viewer)")

    parser.add_argument('--num_neighbors_for_hierarchy', default=4, type=int,
                        help="Number of neighbors to consider when building the Gaussian hierarchy.")
    parser.add_argument('--hierarchy_max_screen_size', type=float, default=2.0,
                        help="Maximum screen size (in pixels) clamped when merging Gaussians into hierarchy nodes")
    parser.add_argument('--hierarchy_screen_size_threshold', type=float, default=0.5,
                        help="Screen size threshold below which Gaussians are candidates for coarsening")
    parser.add_argument('--hierarchy_cam_dist_threshold', type=float, default=1.0,
                        help="Camera distance threshold below which Gaussians are not coarsened")
    parser.add_argument('--hierarchy_recent_kf_skip', type=int, default=50,
                        help="Gaussians placed within this many keyframes are not candidates for coarsening")
    parser.add_argument('--hierarchy_merge_ratio', type=float, default=0.1,
                        help="Minimum fraction of total Gaussians that must be coarsenable to trigger hierarchy creation")
    parser.add_argument('--hierarchy_merge_min_count', type=int, default=200_000,
                        help="Minimum absolute count of coarsenable Gaussians to trigger hierarchy creation")


    ## Depth alignment
    parser.add_argument('--depth_edge_conf_variance', type=float, default=0.2,
                        help="Variance parameter for monocular depth edge confidence (lower = sharper falloff near edges)")
    parser.add_argument('--depth_align_outlier_mult', type=float, default=5.0,
                        help="Error deviation multiplier for outlier rejection during robust depth alignment")
    parser.add_argument('--depth_grid_outlier_mult', type=float, default=7.0,
                        help="Error deviation multiplier for outlier rejection during grid-based depth alignment")
    parser.add_argument('--depth_grid_size', type=int, default=4,
                        help="Grid resolution (NxN) for spatially-varying depth alignment (0 to disable grid alignment)")
    parser.add_argument('--depth_grid_min_pts_per_block', type=int, default=10,
                        help="Minimum points per grid block to use local depth alignment scale")
    parser.add_argument('--depth_grid_min_kpts_per_block', type=int, default=30,
                        help="Minimum keypoints per grid block for valid block-level depth alignment")
    parser.add_argument('--depth_grid_scale_tolerance', type=float, default=0.5,
                        help="Maximum fractional deviation of per-block scale from global scale")

    ## Training proba update
    parser.add_argument('--training_proba_update_freq', type=int, default=50,
                        help="Update training probabilities and offload inactive keyframes every this many keyframes")

    ## Keyframe management
    parser.add_argument('--max_active_keyframes', type=int, default=250,
                        help="Maximum number of keyframes to keep in GPU memory. Will start offloading keyframes to CPU if this number is exceeded.")
    parser.add_argument('--num_closest_active_keyframes', type=int, default=50,
                        help="Number of spatially closest keyframes to the last frame that are always kept active for training")
    parser.add_argument('--min_training_proba', type=float, default=0.1,
                        help="Minimum training probability threshold below which keyframes are offloaded to CPU")
    parser.add_argument('--gpu_mem_clear_fraction', type=float, default=0.95,
                        help="Fraction of total GPU memory in use (from the driver's free/total, so other processes count too) above which gc.collect() and torch.cuda.empty_cache() are forced")

    ## Evaluation
    parser.add_argument('--test_hold', type=int, default=-1, 
                        help="Holdout for test set, will exclude every test_hold image from the Gaussian optimization and use them for testing. The test frames will still be used for training the pose. If set to -1, no keyframes will be excluded from training.")
    parser.add_argument('--test_frequency', type=int, default=-1, 
                        help="Test and get metrics every test_frequency keyframes")
    parser.add_argument('--display_runtimes', action='store_true',
                        help="Display runtimes for each step in the tqdm bar")

    ## Checkpoint options
    parser.add_argument('-m', '--model_path', default="", 
                        help="Directory to store the renders from test view and checkpoints after training. If not set, will be set to results/xxxxxx.")
    parser.add_argument('--save_every', default=-1, type=int, 
                        help="Frequency of saving an intermediate model w.r.t input frames.")
    parser.add_argument('--save_keyframes', action='store_true', 
                        help="Save keyframe images to model_path/images")
    parser.add_argument('--cglf_scene_path', type=str, default="",
                        help="If set, export the final registered images, COLMAP cameras, and filtered BA landmarks as a new CGLF scene directory. Existing directories are never overwritten.")

    ## Viewer
    parser.add_argument('--viewer_mode', choices=['local', 'server', 'none'], default='none')
    parser.add_argument('--ip', type=str, default="0.0.0.0",
                        help="Interface for the viewer server to listen on, if using server viewer_mode. 0.0.0.0 allows remote viewer clients, use 127.0.0.1 to restrict to local connections.")
    parser.add_argument('--port', type=int, default=6009,
                        help="Port for the viewer server to listen on, if using server viewer_mode")
    parser.add_argument('--keep_alive', action='store_true',
                        help="Keep the viewer alive after training completes")

    args = parser.parse_args()

    ## Set the output directory if not specified
    if args.model_path == "":
        i = 0
        while os.path.exists(f"results/{i:06d}"):
            i += 1
        args.model_path = f"results/{i:06d}"

    return args
