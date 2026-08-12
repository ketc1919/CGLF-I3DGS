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

from argparse import Namespace
from typing import List
import gc
import os
import json
import math
import tempfile
import threading
import time
import torch
import torch.nn.functional as F

from poses.feature_detector import DescribedKeypoints
from poses.matcher import Matcher
from poses.guided_mvs import GuidedMVS
from poses.ba_problem import BAProblem
from scene.mono_depth import MonoDepthEstimator
from scene.keyframe import Keyframe
from scene.optimizers import BaseAdam, SparseGaussianAdam
from utils import (
    focal2fov,
    fov2focal,
    free_cuda_memory,
    getProjectionMatrix,
    align_poses_against_first_n,
    batch_write_imgs,
)
from dataloaders.read_write_model import write_model
from scene.hierarchy import HierarchyStructure

from scene.scene_model_renderer import RenderingMixin
from scene.scene_model_trainer import GaussianTrainerMixin
from scene.scene_model_loop_closure import LoopClosureMixin
from scene.scene_model_evaluation import EvaluationMixin

DEFAULT_VIEWER_WIDTH = 1280
DEFAULT_VIEWER_HEIGHT = 720
DEFAULT_VIEWER_FOVY_DEG = 60.0


class SceneModel(RenderingMixin, GaussianTrainerMixin, LoopClosureMixin, EvaluationMixin):
    """
    Scene Model class that contains the scene's Gaussians, keyframes, and methods for rendering and optimization.

    Functionality is organized into mixins:
        - RenderingMixin (scene/scene_model_renderer.py): Gaussian rendering and rasterization
        - GaussianTrainerMixin (scene/scene_model_trainer.py): Training loop, Gaussian initialization, and hierarchy management
        - LoopClosureMixin (scene/scene_model_loop_closure.py): Loop closure, bundle adjustment, and keyframe matching
        - EvaluationMixin (scene/scene_model_evaluation.py): Metrics evaluation, test rendering, and exposure harmonization
    """
    def __init__(
        self,
        width: int,
        height: int,
        args: Namespace,
        matcher: Matcher = None,
        inference_mode: bool = False,
    ):
        """
        Args:
            width: Width of the image.
            height: Height of the image.
            args: Arguments for the scene model. Training parameters required if inference_mode is False.
            matcher: Matcher for the scene model. Defaults to None (if inference_mode is True).
            inference_mode: Whether we load the scene for visualization. Defaults to False.
        """
        self.width = width
        self.height = height
        self.matcher = matcher
        self.centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device="cuda")
        self.optimization_thread = None
        # Dedicated RNG for the (concurrent, non-deterministic) Gaussian optimization so poses stay reproducible under --deterministic_poses.
        self.gs_rng = torch.Generator(device="cuda").manual_seed(0)
        self.args = args
        self.n_vpr_slots = 1
        self.render_tau = getattr(args, 'render_tau', 1)
        self.f = torch.tensor(0.7 * width, device="cuda")
        # Reentrant: the optimization step holds it across forward+backward while
        # render() re-acquires it for the forward pass.
        self.lock = threading.RLock()
        self.latest_keyframe_time = time.perf_counter()
        s = (height**2 + width**2)**0.5
        self.max_scale_screen_size = getattr(args, 'prune_max_screen_size', 0.05) * s

        self.lpips = None

        if not inference_mode:
            self.num_neighbors_for_hierarchy = args.num_neighbors_for_hierarchy
            assert self.num_neighbors_for_hierarchy == 4, "Currently only 4 neighbors for hierarchy supported"
            self.num_prev_keyframes_check = args.num_prev_keyframes_check
            self.active_sh_degree = args.sh_degree
            self.max_sh_degree = args.sh_degree
            self.lambda_dssim = args.lambda_dssim
            self.init_proba_scaler = args.init_proba_scaler
            self.max_active_keyframes = args.max_active_keyframes
            self.use_last_frame_proba = args.use_last_frame_proba
            self.guided_mvs = GuidedMVS(args)
            self.depth_estimator = MonoDepthEstimator(
                width, height, edge_conf_variance=args.depth_edge_conf_variance, batch=1,
            )
            max_obs = 2 * args.num_prev_keyframes_miniba_incr + 1
            self.num_keyframes_localba = args.num_keyframes_localba
            self.min_loop_size = args.min_loop_size
            self.save_keyframes = args.save_keyframes
            self.ba_outlier_threshold = max(args.ba_outlier_threshold * s, 1.0)
            self.ba_problem = BAProblem(args, max_obs, self.ba_outlier_threshold)
            self.training_proba = torch.empty(0, device="cuda")

            ## Initialize HierarchyStructure
            self.hierarchy = HierarchyStructure(args, self.lock, inference_mode)
            self.update_f(self.f)
            self.autograd_dummy = torch.zeros(1, requires_grad=True, device="cuda")
            self._init_gaussian_optimizer()

        self._init_shared_state(inference_mode)

    def _init_gaussian_optimizer(self):
        """Build the SparseGaussianAdam over the gaussian primitives."""
        self.gaussian_optimizer = SparseGaussianAdam(
            {key: self.hierarchy.active_gaussians[key]
             for key in self.hierarchy.PRIMITIVE_KEYS},
            betas=(self.args.adam_beta1, self.args.adam_beta2),
        )

    def _init_shared_state(self, inference_mode: bool = False):
        self.keyframes: List[Keyframe] = []

        self.prev_keyframes_odo = None
        self.prev_keyframes_loop = None
        self.vpr_features = None
        self.approx_cam_centres = None
        self.are_test = torch.empty(0, device="cuda", dtype=torch.bool)
        self.gt_Rts = torch.empty(0, 4, 4, device="cuda")
        self.gt_Rts_mask = torch.empty(0, device="cuda", dtype=torch.bool)
        self.gt_f = self.f
        self.cached_Rts = torch.empty(0, 4, 4, device="cuda")
        self.valid_Rt_cache = torch.empty(0, device="cuda", dtype=torch.bool)
        self.sorted_frame_indices = None
        self.last_trained_id = 0
        self.valid_keyframes = torch.empty(0, dtype=torch.bool)
        self.inference_mode = inference_mode

        self._init_pixel_helpers(self.width, self.height)

    def reset(
        self,
        inference_mode: bool = False,
    ):
        self.centre = torch.tensor([(self.width - 1) / 2, (self.height - 1) / 2], device="cuda")
        self.optimization_thread = None
        self.f = torch.tensor(0.7 * self.width, device="cuda")

        if not inference_mode:
            self.num_neighbors_for_hierarchy = self.args.num_neighbors_for_hierarchy
            self.active_sh_degree = self.args.sh_degree
            self.max_sh_degree = self.args.sh_degree
            if hasattr(self, "ba_problem"):
                self.ba_problem.reset(self.args, max_obs = 2 * self.args.num_prev_keyframes_miniba_incr + 1, ba_outlier_threshold=self.ba_outlier_threshold)
            else:
                self.ba_problem = BAProblem(self.args, max_obs = 2 * self.args.num_prev_keyframes_miniba_incr + 1, ba_outlier_threshold=self.ba_outlier_threshold)
            self.training_proba = torch.empty(0, device="cuda")

            ## Initialize HierarchyStructure
            self.hierarchy.reset(self.args, inference_mode)
            self.update_f(self.f)
            self.autograd_dummy = torch.zeros(1, requires_grad=True, device="cuda")
            self.adam_step_count = 1
            self._init_gaussian_optimizer()

        self._init_shared_state(inference_mode)

        free_cuda_memory()

    def update_f(self, f):
        self.f = f
        self.hierarchy.f = f
        self.init_intrinsics()

    @property
    def gaussian_params(self):
        return self.hierarchy.active_gaussians

    def _scene_mutation_lock(self):
        """Serialise main-thread scene mutations (BA / loop closure) against the
        optimization step while autograd-tracked primitives/poses are live."""
        return self.lock

    @property
    def xyz(self):
        return self.hierarchy.xyz

    @property
    def f_dc(self):
        return self.hierarchy.f_dc

    @property
    def f_rest(self):
        return self.hierarchy.f_rest

    @property
    def log_scaling(self):
        return self.hierarchy.log_scaling

    @property
    def scaling(self):
        return self.hierarchy.scaling

    def get_rotation(self, mask):
        return self.hierarchy.get_rotation(mask)

    @property
    def rotation(self):
        return self.hierarchy.rotation

    @property
    def opacity(self):
        return self.hierarchy.opacity

    @property
    def kf_id(self):
        return self.hierarchy.kf_id

    @property
    def children(self):
        return self.hierarchy.children

    @property
    def parent(self):
        return self.hierarchy.parent

    @property
    def id_in_hierarchy(self):
        return self.hierarchy.id_in_hierarchy

    @property
    def n_active_gaussians(self):
        return self.hierarchy.n_active_gaussians

    @classmethod
    def _get_metadata_path(cls, scene_source: str):
        if os.path.isdir(scene_source):
            metadata_path = os.path.join(scene_source, "metadata.json")
        else:
            metadata_path = os.path.join(os.path.dirname(scene_source), "metadata.json")
        return metadata_path if os.path.isfile(metadata_path) else None

    @classmethod
    def _load_metadata_for_source(cls, scene_source: str):
        metadata_path = cls._get_metadata_path(scene_source)
        if metadata_path is None:
            return None
        with open(metadata_path) as f:
            return json.load(f)

    @classmethod
    def from_scene(cls, scene_source: str, args):
        metadata = cls._load_metadata_for_source(scene_source)
        if metadata is not None:
            width = metadata["config"]["width"]
            height = metadata["config"]["height"]
        else:
            width = DEFAULT_VIEWER_WIDTH
            height = DEFAULT_VIEWER_HEIGHT
        scene_model = cls(width, height, args, inference_mode=True)
        scene_model.load_scene(scene_source, metadata=metadata)
        return scene_model

    def _resolve_scene_source(self, scene_source: str):
        if os.path.isdir(scene_source):
            candidates = [
                os.path.join(scene_source, "hierarchy.ply"),
                os.path.join(scene_source, "hierarchy_leaves.ply"),
                os.path.join(scene_source, "hierarchy_leaves.spz"),
            ]
            for candidate in candidates:
                if os.path.isfile(candidate):
                    return candidate
            raise FileNotFoundError(
                f"No supported Gaussian scene file found in {scene_source}. "
                "Expected one of hierarchy.ply, hierarchy_leaves.ply, or hierarchy_leaves.spz."
            )

        if os.path.isfile(scene_source):
            suffix = os.path.splitext(scene_source)[1].lower()
            if suffix in {".ply", ".spz"}:
                return scene_source

        raise FileNotFoundError(
            f"Unsupported scene source: {scene_source}. "
            "Expected a scene directory, a .ply file, or a .spz file."
        )

    def _load_hierarchy_from_file(self, source_path: str, sh_degree=None, num_neighbors_for_hierarchy=None):
        hierarchy_kwargs = dict(
            sh_degree=sh_degree,
            num_neighbors_for_hierarchy=num_neighbors_for_hierarchy,
            hierarchy_max_screen_size=getattr(self.args, 'hierarchy_max_screen_size', 2.0),
            hierarchy_screen_size_threshold=getattr(self.args, 'hierarchy_screen_size_threshold', 0.5),
            hierarchy_cam_dist_threshold=getattr(self.args, 'hierarchy_cam_dist_threshold', 1.0),
            hierarchy_recent_kf_skip=getattr(self.args, 'hierarchy_recent_kf_skip', 50),
            hierarchy_merge_ratio=getattr(self.args, 'hierarchy_merge_ratio', 0.1),
            hierarchy_merge_min_count=getattr(self.args, 'hierarchy_merge_min_count', 200_000),
        )

        if os.path.splitext(source_path)[1].lower() == ".spz":
            temp_ply_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as temp_file:
                    temp_ply_path = temp_file.name
                convert_any_splat_to_ply(source_path, temp_ply_path)
                return HierarchyStructure.from_ply(temp_ply_path, **hierarchy_kwargs)
            finally:
                if temp_ply_path is not None and os.path.exists(temp_ply_path):
                    os.unlink(temp_ply_path)

        return HierarchyStructure.from_ply(source_path, **hierarchy_kwargs)

    def _infer_loaded_sh_degree(self):
        return int(math.isqrt(self.hierarchy.gaussian_nodes["f_rest"]["val"].shape[1] + 1) - 1)

    def load_scene(self, scene_source: str, metadata=None):
        if metadata is None:
            metadata = self._load_metadata_for_source(scene_source)

        self.reset(inference_mode=True)
        resolved_source = self._resolve_scene_source(scene_source)

        self.hierarchy = self._load_hierarchy_from_file(
            resolved_source,
            sh_degree=None,
            num_neighbors_for_hierarchy=metadata["config"].get("num_neighbors_for_hierarchy", 4) if metadata is not None else None,
        )
        self.active_sh_degree = self._infer_loaded_sh_degree()
        self.max_sh_degree = self.active_sh_degree

        if metadata is not None:
            self.update_f(torch.tensor(metadata["config"]["f"], device="cuda"))
        else:
            default_f = fov2focal(math.radians(DEFAULT_VIEWER_FOVY_DEG), self.height)
            self.update_f(torch.tensor(default_f, device="cuda"))
        self.hierarchy.lock = self.lock

        if metadata is None:
            return

        # Load keyframes
        height = metadata["config"]["height"]
        width = metadata["config"]["width"]
        for i in range(len(metadata["keyframes"])):
            keyframe = Keyframe.from_json(metadata["keyframes"][i], i, height, width)
            self.add_keyframe(keyframe)

    @property
    def n_active_keyframes(self):
        return len(self.keyframes)

    def get_Rts(self):
        invalid_ids = torch.where(~self.valid_Rt_cache)[0]
        if len(invalid_ids) > 0:
            for keyframe_id in invalid_ids:
                self.cached_Rts[keyframe_id] = self.keyframes[keyframe_id].get_Rt()
            self.valid_Rt_cache[invalid_ids] = True
        return self.cached_Rts

    def get_keyframe_Rts(self, keyframe_ids=None):
        Rts = self.get_Rts()
        n_rts = Rts.shape[0]
        if keyframe_ids is None:
            # Cap at n_rts so concurrent viewer reads never request a pose row
            # that has not been published yet.
            keyframe_ids = torch.arange(min(len(self.keyframes), n_rts), device=Rts.device)
        elif not torch.is_tensor(keyframe_ids):
            keyframe_ids = torch.tensor(keyframe_ids, device=Rts.device, dtype=torch.long)
        else:
            keyframe_ids = keyframe_ids.to(device=Rts.device, dtype=torch.long)
        keyframe_ids = keyframe_ids[(keyframe_ids >= 0) & (keyframe_ids < n_rts)]
        if keyframe_ids.numel() == 0:
            return Rts[:0]
        return Rts[keyframe_ids]

    def get_gt_Rts(self, align, align_n = 0):
        n_poses = min(self.gt_Rts_mask.shape[0], self.cached_Rts.shape[0])
        if align and n_poses > 0:
            Rts = self.get_Rts()[:n_poses][self.gt_Rts_mask[:n_poses]]
            return align_poses_against_first_n(self.gt_Rts[: len(Rts)], Rts, align_n)
        else:
            return self.gt_Rts

    def get_gt_keyframe_Rts(self, align, align_n=0):
        gt_Rts = self.get_gt_Rts(align, align_n)
        if len(gt_Rts) == 0:
            return gt_Rts
        n_poses = min(self.gt_Rts_mask.shape[0], self.cached_Rts.shape[0])
        if align:
            keyframe_ids = torch.where(self.gt_Rts_mask[:n_poses])[0][:len(gt_Rts)]
        else:
            keyframe_ids = torch.arange(min(len(gt_Rts), n_poses), device=gt_Rts.device)
        return gt_Rts[:len(keyframe_ids)]

    def init_intrinsics(self):
        self.FoVx = focal2fov(self.f, self.width)
        self.FoVy = focal2fov(self.f, self.height)
        self.tanfovx = math.tan(self.FoVx * 0.5)
        self.tanfovy = math.tan(self.FoVy * 0.5)
        self.projection_matrix = (
            getProjectionMatrix(
                znear=getattr(self.args, "render_near_plane", 0.01),
                zfar=getattr(self.args, "render_far_plane", 10000.0),
                fovX=self.FoVx, fovY=self.FoVy,
            )
            .transpose(0, 1)
            .cuda()
        )

    def add_keyframe(self, keyframe: Keyframe, vpr_feature: torch.Tensor=None, f=None):
        """Add a keyframe to the scene, add and prune Gaussians"""

        # Make sure training is not running
        self.join_optimization_thread()

        ## Add the keyframe data, then publish it in self.keyframes at the end.
        # The viewer reads len(self.keyframes) concurrently; publishing the list
        # entry before growing cached_Rts lets it request an out-of-range pose.
        self.latest_keyframe_time = keyframe.info.get("source_frame_time", time.perf_counter())
        self.are_test = torch.cat([self.are_test, torch.tensor([keyframe.info["is_test"]], device=self.are_test.device, dtype=self.are_test.dtype)])
        if vpr_feature is not None:
            slots = vpr_feature[:self.n_vpr_slots].contiguous().clone()[None]   # [1, S, D]
            if self.vpr_features is None:
                self.vpr_features = slots
            else:
                self.vpr_features = torch.cat([self.vpr_features, slots], dim=0)

        if self.approx_cam_centres is None:
            self.approx_cam_centres = keyframe.approx_centre[None]
        else:
            self.approx_cam_centres = torch.cat(
                [self.approx_cam_centres, keyframe.approx_centre[None]], dim=0
            )
        ## Update intrinsics
        if f is not None:
            self.update_f(f)

        ## Update cached Rts for the viewer
        Rt = keyframe.get_Rt()
        self.cached_Rts = torch.cat(
            [self.cached_Rts, Rt.unsqueeze(0)], dim=0
        )
        self.valid_Rt_cache = torch.cat(
            [self.valid_Rt_cache, torch.ones(1, device="cuda", dtype=torch.bool)], dim=0
        )
        gt_pose = keyframe.info.get("Rt", None)
        if gt_pose is not None:
            self.gt_Rts = torch.cat([self.gt_Rts, gt_pose.unsqueeze(0)], dim=0)
        self.gt_Rts_mask = torch.cat(
            [
                self.gt_Rts_mask,
                torch.tensor([gt_pose is not None], device=self.gt_Rts_mask.device, dtype=self.gt_Rts_mask.dtype),
            ],
            dim=0,
        )
        self.gt_f = keyframe.info.get("focal", self.f)

        self.keyframes.append(keyframe)

        ## Periodically empty cache
        if not self.inference_mode:
            keyframe.desc_kpts.full_feats = None
            n_keyframes = len(self.keyframes)
            if n_keyframes % self.args.training_proba_update_freq == 0 and n_keyframes > self.max_active_keyframes:
                self.update_training_proba()
                gc.collect()
            else:
                self.training_proba = torch.cat(
                    [self.training_proba, torch.ones(1, device="cuda") * (not keyframe.is_test)])

    def enable_inference_mode(self):
        """Enable inference mode."""
        with self.lock:
            self.inference_mode = True
            self.hierarchy.enable_inference_mode()

    def save(self, path: str, reconstruction_time: float = 0, n_frames: int = 0):
        print("max_gpu_allocated_gb: ", torch.cuda.max_memory_allocated() / 1024**3)
        print("max_n_active_gaussians: ", getattr(self, 'max_n_active_gaussians', 0))

        # Get metrics
        metrics = {
            "num keyframes": len(self.keyframes),
        }
        if reconstruction_time > 0:
            metrics["time"] = reconstruction_time
            if n_frames > 0:
                metrics["FPS"] = n_frames / reconstruction_time
        metrics.update(self.evaluate(True, True, True, os.path.join(path, "test_images")))

        if path == "":
            print("No path provided, skipping save")
            return metrics

        hierarchy_path = os.path.join(path, "hierarchy.ply")
        os.makedirs(path, exist_ok=True)
        hierarchy_saved = False
        try:
            self.hierarchy.save_ply(hierarchy_path)
            hierarchy_saved = True
        except Exception as e:
            print(e)
            print("Saving hierarchy failed")

        # Save metadata
        metadata = {
            "config": {
                "width": self.width,
                "height": self.height,
                "sh_degree": self.max_sh_degree,
                "num_neighbors_for_hierarchy": self.hierarchy.num_neighbors_for_hierarchy,
                "f": self.f.item(),
            },
            "keyframes": [keyframe.to_json() for keyframe in self.keyframes],
        }
        metadata = {**metrics, **metadata}

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        # Saving cameras with COLMAP format
        images = {}
        cameras = {}
        colmap_save_path = os.path.join(path, "sparse/0")
        os.makedirs(colmap_save_path, exist_ok=True)
        next_colmap_id = 0
        for index, keyframe in enumerate(self.keyframes):
            kf_cameras, kf_images = keyframe.to_colmap(index, first_colmap_id=next_colmap_id)
            for camera, image in zip(kf_cameras, kf_images):
                cameras[camera.id] = camera
                images[image.id] = image
            next_colmap_id += len(kf_images)
        write_model(cameras, images, {}, colmap_save_path, ext=".bin")

        # Save the keyframes.
        if self.save_keyframes:
            imgs = [keyframe.image for keyframe in self.keyframes]
            img_paths = [os.path.join(path, "images", f"{keyframe.index:06d}.jpeg") for keyframe in self.keyframes]
            batch_write_imgs(imgs, img_paths)

            masked_keyframes = [kf for kf in self.keyframes if kf.mask is not None]
            if masked_keyframes:
                masks = [kf.mask.float() for kf in masked_keyframes]
                mask_paths = [os.path.join(path, "masks", f"{kf.index:06d}.png") for kf in masked_keyframes]
                batch_write_imgs(masks, mask_paths)

        return metrics

    def get_closest_keyframe(
        self, position: torch.Tensor, count: int = 1
    ) -> list[Keyframe]:
        dists = torch.linalg.vector_norm(
            self.approx_cam_centres - position[None], dim=-1
        )
        closest_ids = dists.argsort()[:count]
        return [self.keyframes[closest_id] for closest_id in closest_ids]
