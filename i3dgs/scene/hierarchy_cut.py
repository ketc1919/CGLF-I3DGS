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

import gc
import time
import random
import threading
from typing import Optional

import torch
from threading import Thread

from utils import free_cuda_memory


class HierarchyCutMixin:
    """Mixin providing screen-space cut management (refine/coarsen) for HierarchyStructure."""

    hierarchy_screen_size_threshold: float
    hierarchy_cam_dist_threshold: float
    hierarchy_recent_kf_skip: int
    hierarchy_merge_ratio: float
    hierarchy_merge_min_count: int
    last_hierarchy_creation_kf_index: Optional[int]
    f: torch.Tensor
    lock: threading.Lock
    update_lock: threading.Lock
    update_thread: Optional[Thread]
    inference_mode: bool
    active_gaussians_count: int
    active_gaussians: dict
    gaussian_nodes: dict
    cpu_nodes_count: int
    n_updates: int
    xyz: torch.Tensor
    scaling: torch.Tensor
    parent: torch.Tensor
    kf_id: torch.Tensor
    children: torch.Tensor
    id_in_hierarchy: torch.Tensor

    @torch.no_grad()
    def create_hierarchy_coarser_mask(self, cam_centre, index):
        screen_size_threshold = self.hierarchy_screen_size_threshold
        cam_dist_threshold = self.hierarchy_cam_dist_threshold

        # If Gaussians appear subpixel, coarsen them
        dist = torch.linalg.vector_norm(
            self.xyz - cam_centre[None], dim=-1
        )
        screen_size = self.f * self.scaling.max(dim=-1)[0] / dist
        mask = screen_size < screen_size_threshold

        # Don't coarsen if close to the camera
        mask &= dist > cam_dist_threshold

        # Don't coarsen if they have parent:
        mask &= self.parent == -1

        # Don't coarsen if placed recently
        mask &= (self.kf_id - index).abs() > self.hierarchy_recent_kf_skip

        return mask

    @torch.no_grad()
    def check_and_create_hierarchy(self, keyframe):
        cam_centre = keyframe.get_centre()
        ## Bringing finer Gaussians
        n, k = self._update_cut(cam_centre, 2)

        ## Coarsening
        # Cooldown
        if (
            self.last_hierarchy_creation_kf_index is not None
            and keyframe.index - self.last_hierarchy_creation_kf_index < 50
        ):
            return 0

        mask = self.create_hierarchy_coarser_mask(cam_centre, keyframe.index)

        # Only create the hierarchy if we have a lot of Gaussians to remove
        if mask.sum() > max(len(self.xyz) * self.hierarchy_merge_ratio, self.hierarchy_merge_min_count):
            n_merged = self.create_hierarchy_level(cam_centre, mask)
            if n_merged > 0:
                self.last_hierarchy_creation_kf_index = keyframe.index
            return n_merged
        return 0

    @torch.no_grad()
    def update_cut(self, cam_centre, render_tau):
        if not self.inference_mode:
            return
        if self.update_thread is None or not self.update_thread.is_alive():
            self.update_thread = Thread(target=self._update_cut, args=(cam_centre, render_tau, True))
            self.update_thread.start()

    @torch.no_grad()
    def update_cut_for_test(self, cam_centre, render_tau):
        for _ in range(10):
            n_removed, n_added = self._update_cut(cam_centre, render_tau)
            if max(n_removed, n_added) == 0:
                break
        free_cuda_memory()

    def count_hierarchy_levels(self):
        """Returns the number of levels in the Gaussian hierarchy (1 if no merging has occurred)."""
        n = self.cpu_nodes_count
        if n == 0:
            return 0

        children = self.gaussian_nodes["children"]["val"][:n]  # [N, K], CPU tensor
        parent = self.gaussian_nodes["parent"]["val"][:n]       # [N], CPU tensor

        # Start from leaves (no valid children)
        frontier = (children < 0).all(dim=1).nonzero(as_tuple=True)[0]
        num_levels = 1
        while True:
            parent_ids = parent[frontier]
            frontier = parent_ids[parent_ids >= 0].unique()
            if frontier.numel() == 0:
                break
            num_levels += 1

        return num_levels

    @torch.no_grad()
    def _update_cut(self, cam_centre, render_tau, sleep=False):
        with self.update_lock:
            to_make_finer, children_to_add, _ = self._find_gaussians_to_refine(cam_centre, render_tau)
            parents_to_add, children_to_remove_mask = self._find_gaussians_to_coarsen(render_tau, cam_centre)
            n_removed, n_added = self._apply_cut_update(to_make_finer, children_to_add, parents_to_add, children_to_remove_mask)

        if sleep:
            time.sleep(0.1)

        return n_removed, n_added

    @torch.no_grad()
    def _find_gaussians_to_refine(self, cam_centre, render_tau):
        # Get the screen size of the Gaussians
        dist = torch.linalg.vector_norm(
            self.xyz - cam_centre[None], dim=-1
        )
        screen_size = self.f * self.scaling.mean(dim=-1) / dist

        # Make finer: Replace coarse Gaussians with their children
        coarse_mask = screen_size > render_tau
        has_children = (self.children >= 0).any(dim=-1)
        to_make_finer = coarse_mask & has_children

        # Collect children to load from hierarchy
        children_to_add = self.children[to_make_finer].view(-1)
        children_to_add = children_to_add[children_to_add >= 0]  # Filter invalid

        return to_make_finer, children_to_add, screen_size

    @torch.no_grad()
    def _find_gaussians_to_coarsen(self, render_tau, cam_centre):
        device = self.xyz.device

        # Get the screen size of the Gaussians
        dist = torch.linalg.vector_norm(
            self.xyz - cam_centre[None], dim=-1
        )
        screen_size = self.f * self.scaling.mean(dim=-1) / dist

        # Make coarser: Replace fine Gaussians with their parent
        # Only coarsen if ALL siblings of a parent are also fine
        fine_mask = screen_size < render_tau
        # Don't make coarser if training, that's handled by create_hierarchy_level.
        has_parent = self.parent >= 0
        candidates_for_coarser = fine_mask & has_parent

        # Group by parent and check if all children of that parent are fine
        # Vectorized using bincount
        if candidates_for_coarser.any():
            parent_ids = self.parent

            # Count fine children per parent (only allocates for parents that appear)
            unique_parents, fine_counts = torch.unique(
                parent_ids[candidates_for_coarser], return_counts=True
            )

            # Parents where ALL 4 children are fine
            parents_to_add = unique_parents[fine_counts == 4].to(torch.int32)

            # Mark children to remove - those whose parent is valid
            children_to_remove_mask = torch.isin(parent_ids, parents_to_add) & has_parent
        else:
            parents_to_add = torch.empty(0, dtype=torch.int32, device=device)
            children_to_remove_mask = torch.zeros(self.active_gaussians_count, dtype=torch.bool, device=device)

        # Filter parents for coarsening: only coarsen if parent screen_size < render_tau
        # Do filtering on CPU to save GPU memory
        if parents_to_add.numel() > 0:
            ids_cpu = parents_to_add.cpu()
            parent_xyz = self.gaussian_nodes["xyz"]["val"][ids_cpu]
            parent_scaling = torch.exp(self.gaussian_nodes["scaling"]["val"][ids_cpu])
            cam_centre_cpu = cam_centre.cpu()
            parent_dist = torch.linalg.vector_norm(parent_xyz - cam_centre_cpu[None], dim=-1)
            parent_screen_size = self.f.cpu() * parent_scaling.mean(dim=-1) / parent_dist
            valid_parents_mask = (parent_screen_size < render_tau).to(device)

            # Only keep children_to_remove for valid parents
            invalid_parents = parents_to_add[~valid_parents_mask]
            if invalid_parents.numel() > 0:
                is_child_of_invalid = torch.isin(self.parent, invalid_parents)
                children_to_remove_mask &= ~is_child_of_invalid

            parents_to_add = parents_to_add[valid_parents_mask]

        return parents_to_add, children_to_remove_mask

    @torch.no_grad()
    def _apply_cut_update(self, to_make_finer, children_to_add, parents_to_add, children_to_remove_mask):
        device = self.xyz.device

        # Combine nodes to add
        nodes_to_add = torch.cat([children_to_add.to(torch.int32), parents_to_add.to(torch.int32)])

        # Load new Gaussians from CPU hierarchy (non_blocking for overlap)
        new_gaussians = {}
        if nodes_to_add.numel() > 0:
            ids_cpu = nodes_to_add.cpu()
            for key in self.gaussian_nodes.keys():
                new_gaussians[key] = self.gaussian_nodes[key]["val"][ids_cpu].to(device, non_blocking=True)

        # Determine which Gaussians to keep
        # Remove: Gaussians being made finer + Gaussians being made coarser (grouped by parent)
        to_remove = to_make_finer | children_to_remove_mask
        kept_mask = ~to_remove

        # Move optimized Gaussians to the hierarchy
        if not self.inference_mode:
            ids = self.id_in_hierarchy[to_remove].cpu()
            for key in self.gaussian_nodes.keys():
                val = self.active_gaussians[key]["val"][:self.active_gaussians_count][to_remove]
                self.gaussian_nodes[key]["val"][ids] = val.cpu()

        # Update active Gaussians
        n_removed = to_remove.sum().item()
        n_added = nodes_to_add.numel()

        if n_removed > 0 or n_added > 0:
            with self.lock:
                self.add_and_prune(new_gaussians, kept_mask)

            # Free memory from time to time
            if random.random() < 0.01:
                gc.collect()

            self.n_updates += 1

        return n_removed, n_added
