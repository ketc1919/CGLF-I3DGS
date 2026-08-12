#
# Copyright (C) 2026, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

from typing import List
from collections import deque
import heapq
import math
import torch
import random
from collections import Counter
import cupy

from poses.mini_ba import MiniBA
from scene.keyframe import Keyframe
from utils import mtx2sixD, sixD2mtx, free_cuda_memory


class PoseGraph:
    def __init__(self):
        self.adj = []
        self.weights = []

    def add_match(self, u, v, weight):
        max_id = max(u, v)
        while len(self.adj) <= max_id:
            self.adj.append([])
        edge_id = len(self.weights)
        self.weights.append(weight)
        self.adj[u].append((v, edge_id))
        self.adj[v].append((u, edge_id))

    def _sync_graph(self):
        return

    def vcount(self):
        return len(self.adj)

    def neighbors(self, v):
        if v < 0 or v >= len(self.adj):
            return []
        return [neighbor for neighbor, _ in self.adj[v]]

    def edge_weight(self, u, v):
        if u < 0 or u >= len(self.adj):
            return math.inf
        for neighbor, edge_id in self.adj[u]:
            if neighbor == v:
                return self.weights[edge_id]
        return math.inf

    def _bfs_distances(self, source, targets=None):
        dist = [math.inf] * len(self.adj)
        if source < 0 or source >= len(self.adj):
            return dist
        remaining = set(targets) if targets is not None else None
        dist[source] = 0
        queue = deque([source])
        if remaining is not None:
            remaining.discard(source)
        while queue:
            if remaining is not None and not remaining:
                break
            current = queue.popleft()
            next_dist = dist[current] + 1
            for neighbor, _ in self.adj[current]:
                if dist[neighbor] == math.inf:
                    dist[neighbor] = next_dist
                    if remaining is not None:
                        remaining.discard(neighbor)
                    queue.append(neighbor)
        return dist

    def _dijkstra_distances(self, source, targets=None):
        dist = [math.inf] * len(self.adj)
        if source < 0 or source >= len(self.adj):
            return dist
        remaining = set(targets) if targets is not None else None
        dist[source] = 0.0
        heap = [(0.0, source)]
        while heap:
            current_dist, current = heapq.heappop(heap)
            if current_dist != dist[current]:
                continue
            if remaining is not None:
                remaining.discard(current)
                if not remaining:
                    break
            for neighbor, edge_id in self.adj[current]:
                next_dist = current_dist + self.weights[edge_id]
                if next_dist < dist[neighbor]:
                    dist[neighbor] = next_dist
                    heapq.heappush(heap, (next_dist, neighbor))
        return dist

    def _multi_source_dijkstra_distances(self, sources):
        dist = [math.inf] * len(self.adj)
        heap = []
        for source in sources:
            if 0 <= source < len(self.adj) and dist[source] > 0.0:
                dist[source] = 0.0
                heap.append((0.0, source))
        heapq.heapify(heap)
        while heap:
            current_dist, current = heapq.heappop(heap)
            if current_dist != dist[current]:
                continue
            for neighbor, edge_id in self.adj[current]:
                next_dist = current_dist + self.weights[edge_id]
                if next_dist < dist[neighbor]:
                    dist[neighbor] = next_dist
                    heapq.heappush(heap, (next_dist, neighbor))
        return dist

    def _dijkstra_path(self, source, target):
        n = len(self.adj)
        dist = [math.inf] * n
        prev_node = [-1] * n
        prev_edge = [-1] * n
        if source < 0 or source >= n or target < 0 or target >= n:
            return [], []
        dist[source] = 0.0
        heap = [(0.0, source)]
        while heap:
            current_dist, current = heapq.heappop(heap)
            if current_dist != dist[current]:
                continue
            if current == target:
                break
            for neighbor, edge_id in self.adj[current]:
                next_dist = current_dist + self.weights[edge_id]
                if next_dist < dist[neighbor]:
                    dist[neighbor] = next_dist
                    prev_node[neighbor] = current
                    prev_edge[neighbor] = edge_id
                    heapq.heappush(heap, (next_dist, neighbor))
        if dist[target] == math.inf:
            return [], []
        path = []
        edge_ids = []
        current = target
        while current != -1:
            path.append(current)
            edge_id = prev_edge[current]
            if edge_id != -1:
                edge_ids.append(edge_id)
            current = prev_node[current]
        path.reverse()
        edge_ids.reverse()
        return path, [self.weights[edge_id] for edge_id in edge_ids]

    def get_dists(self, v0, v1, w_weights=False):
        distance_fn = self._dijkstra_distances if w_weights else self._bfs_distances
        targets = list(v1)
        target_set = set(targets)
        return [
            [dist[target] if 0 <= target < len(dist) else math.inf for target in targets]
            for dist in (distance_fn(source, target_set) for source in v0)
        ]

    def get_min_dists(self, sources, targets, w_weights=False):
        targets = list(targets)
        if w_weights:
            dist = self._multi_source_dijkstra_distances(sources)
        else:
            queue = deque()
            dist = [math.inf] * len(self.adj)
            for source in sources:
                if 0 <= source < len(self.adj) and dist[source] != 0:
                    dist[source] = 0
                    queue.append(source)
            remaining = set(targets)
            for source in sources:
                remaining.discard(source)
            while queue:
                if not remaining:
                    break
                current = queue.popleft()
                next_dist = dist[current] + 1
                for neighbor, _ in self.adj[current]:
                    if dist[neighbor] == math.inf:
                        dist[neighbor] = next_dist
                        remaining.discard(neighbor)
                        queue.append(neighbor)
        return [dist[target] if 0 <= target < len(dist) else math.inf for target in targets]

    def get_backbone_path(self, head, tail):
        return self._dijkstra_path(head, tail)
    
    def get_backbone_ownership_w(self, backbone_nodes):
        """
        Assigns nodes to nearest backbone node using weighted distances.
        """
        self._sync_graph()
        
        # Priority queue: (distance, current_node, owner)
        pq = [(0, node, node) for node in backbone_nodes]
        heapq.heapify(pq)
        
        node_to_owner = {}
        ownership = {node: [] for node in backbone_nodes}
        
        while pq:
            dist, current_node, current_owner = heapq.heappop(pq)
            
            if current_node in node_to_owner:
                continue
                
            node_to_owner[current_node] = current_owner
            ownership[current_owner].append(current_node)
            
            for neighbor, edge_id in self.adj[current_node]:
                if neighbor not in node_to_owner:
                    weight = self.weights[edge_id]
                    heapq.heappush(pq, (dist + weight, neighbor, current_owner))
                    
        return ownership
    
    def get_most_connected(self, v):
        """
        Finds the node in list 'v' that has the most connections 
        to other nodes within 'v'.
        
        :param v: A list of node indices (integers)
        :return: The index of the most connected node, or None if v is empty.
        """
        node_set = set(v)
        best_node = None
        best_degree = -1
        for node in sorted(v):
            degree = sum(1 for neighbor, _ in self.adj[node] if neighbor in node_set)
            if degree > best_degree:
                best_degree = degree
                best_node = node

        return best_node

class BAProblem:
    def __init__(self, args, max_obs, ba_outlier_threshold):
        # Initialize fields/structures
        self.init_internal(args, max_obs, ba_outlier_threshold)

        # Huber robust loss (down-weights large sub-threshold residuals).
        huber_delta = self.huber_delta_ratio * self.ba_outlier_threshold

        # Initialize the optimizers
        if self.opt_f_until > 0:
            self.miniBA_local_f = MiniBA(
                self.num_keyframes_localba,
                self.num_pts_localba,
                max_obs,
                optimize_focal=True,
                optimize_3Dpts=True,
                make_cuda_graph=True,
                outlier_threshold=self.ba_outlier_threshold,
                iters=args.iters_localba,
                huber_delta=huber_delta,
                outlier_mad_scale=self.localba_outlier_mad_scale,
            )
        self.miniBA_local = MiniBA(
            self.num_keyframes_localba,
            self.num_pts_localba,
            max_obs,
            optimize_focal=False,
            optimize_3Dpts=True,
            make_cuda_graph=True,
            outlier_threshold=self.ba_outlier_threshold,
            iters=args.iters_localba,
            huber_delta=huber_delta,
            outlier_mad_scale=self.localba_outlier_mad_scale,
        )

    def init_internal(self, args, max_obs, ba_outlier_threshold):
        self.num_keyframes_localba = args.num_keyframes_localba
        self.localba_fixed_ratio = args.localba_fixed_ratio
        self.num_pts_localba = args.num_pts_localba
        self.opt_f_until = args.opt_f_until * (not args.fix_focal)
        self.ba_outlier_threshold = ba_outlier_threshold
        self.max_obs = max_obs
        self.min_loop_size = args.min_loop_size
        self.min_matches_edge = args.min_matches_edge
        self.ba_min_pts_per_keyframe = args.ba_min_pts_per_keyframe
        self.global_ba_pts_per_keyframe = args.global_ba_pts_per_keyframe
        self.huber_delta_ratio = getattr(args, "huber_delta_ratio", 0.0)
        self.localba_outlier_mad_scale = getattr(args, "localba_outlier_mad_scale", -1.0)
        self.capacity = 4096
        self.size = 0  # Actual number of landmarks
        self.pose_graph = PoseGraph()

        # Pre-allocate landmark tensors
        self.landmarks     = torch.empty([self.capacity, 3], device="cuda")
        self.lm_primary_kf = torch.full([self.capacity], -1, dtype=torch.int32, device="cuda")
        self.n_obs         = torch.zeros([self.capacity], dtype=torch.int32, device="cuda")

        # COO observation storage (flat arrays, no max_obs cap at append time)
        self.obs_cap = self.capacity * 4
        self.obs_size = 0
        self.obs_lm_ids      = torch.empty([self.obs_cap], dtype=torch.int32, device="cuda")
        self.obs_kf_ids      = torch.empty([self.obs_cap], dtype=torch.int32, device="cuda")
        self.obs_pt2d_ids    = torch.empty([self.obs_cap], dtype=torch.int32, device="cuda")
        self.obs_uvs         = torch.empty([self.obs_cap, 2], device="cuda")

        # GPU-side atomic counters for the CUDA kernels (avoid host↔device sync per call)
        self._obs_size_gpu = torch.zeros(1, dtype=torch.int32, device="cuda")
        self._lm_size_gpu  = torch.zeros(1, dtype=torch.int32, device="cuda")

        # Scratch buffers for _build_dense_obs (avoid per-call cudaMalloc)
        self._lm_to_local_scratch  = torch.full([self.capacity], -1, dtype=torch.int32, device="cuda")
        self._slot_counters_scratch = torch.zeros([self.num_pts_localba + 1], dtype=torch.int32, device="cuda")

        # Load fused append + build_dense_obs kernels
        with open("poses/ba_obs_kernel.cu", "r") as f:
            _cuda_src = f.read()
        _obs_module = cupy.RawModule(code=_cuda_src, options=("--std=c++11",))
        self._add_obs_kernel            = _obs_module.get_function("add_obs_to_existing_kernel")
        self._add_obs_from_other_kernel = _obs_module.get_function("add_obs_from_other_kf_kernel")
        self._build_dense_obs_kernel    = _obs_module.get_function("build_dense_obs_kernel")

    def reset(self, args, max_obs, ba_outlier_threshold):
        self.init_internal(args, max_obs, ba_outlier_threshold)
        free_cuda_memory()


    def _ensure_capacity(self, needed: int):
        """Grow landmark tensors if needed."""
        if self.size + needed <= self.capacity:
            return
        new_capacity = max(int(1.5 * self.capacity), self.size + needed)

        new_landmarks = torch.empty([new_capacity, 3], device="cuda")
        new_landmarks[:self.size] = self.landmarks[:self.size]
        self.landmarks = new_landmarks

        new_primary_kf = torch.full([new_capacity], -1, dtype=torch.int32, device="cuda")
        new_primary_kf[:self.size] = self.lm_primary_kf[:self.size]
        self.lm_primary_kf = new_primary_kf

        new_n_obs = torch.zeros([new_capacity], dtype=torch.int32, device="cuda")
        new_n_obs[:self.size] = self.n_obs[:self.size]
        self.n_obs = new_n_obs

        self._lm_to_local_scratch  = torch.full([new_capacity], -1, dtype=torch.int32, device="cuda")

        self.capacity = new_capacity

    def _ensure_obs_capacity(self, needed: int):
        """Grow flat observation arrays if needed."""
        if self.obs_size + needed <= self.obs_cap:
            return
        new_cap = max(int(1.5 * self.obs_cap), self.obs_size + needed)

        new_lm  = torch.empty([new_cap], dtype=torch.int32, device="cuda")
        new_kf  = torch.empty([new_cap], dtype=torch.int32, device="cuda")
        new_pt  = torch.empty([new_cap], dtype=torch.int32, device="cuda")
        new_uv  = torch.empty([new_cap, 2], device="cuda")

        new_lm[:self.obs_size] = self.obs_lm_ids[:self.obs_size]
        new_kf[:self.obs_size] = self.obs_kf_ids[:self.obs_size]
        new_pt[:self.obs_size] = self.obs_pt2d_ids[:self.obs_size]
        new_uv[:self.obs_size] = self.obs_uvs[:self.obs_size]

        self.obs_lm_ids   = new_lm
        self.obs_kf_ids   = new_kf
        self.obs_pt2d_ids = new_pt
        self.obs_uvs      = new_uv
        self.obs_cap      = new_cap

    def _add_observations_to_existing(
        self,
        match,
        existing_mask,
        keyframes: List[Keyframe],
        kf_id_other: int,
        desc_kpts,
        keyframe: Keyframe,
        Rt_curr_3x4,
        f_curr,
    ):
        """Add observations from current keyframe to existing landmarks (found via other keyframe).

        An observation is added only if reprojecting the existing landmark into the
        current keyframe gives an error < self.ba_outlier_threshold (pixels).
        Keypoints that fail the gate are left unclaimed so that a later match (or the
        fresh-landmark path in append()) can use them.

        Modifies existing_mask in-place to mark keypoints that were added.
        """
        n_match = match.idx.shape[0]
        if n_match == 0:
            return

        # Conservative capacity reserve: at most n_match new obs
        self._ensure_obs_capacity(n_match)

        # Load GPU counter
        self._obs_size_gpu[0] = self.obs_size

        id_in_ba_other = keyframes[kf_id_other].desc_kpts.id_in_ba_problem.cuda()

        n_threads = 256
        grid  = ((n_match + n_threads - 1) // n_threads,)
        block = (n_threads,)
        cupy_stream = cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
        with cupy_stream:
            self._add_obs_kernel(
                grid=grid, block=block,
                args=(
                    n_match,
                    match.idx.data_ptr(),
                    match.idx_other.data_ptr(),
                    id_in_ba_other.data_ptr(),
                    existing_mask.data_ptr(),
                    keyframe.index,
                    desc_kpts.kpts.data_ptr(),
                    Rt_curr_3x4.data_ptr(),
                    f_curr.data_ptr(),
                    keyframe.centre.data_ptr(),
                    cupy.float32(self.ba_outlier_threshold),
                    self.landmarks.data_ptr(),
                    self.obs_lm_ids.data_ptr(),
                    self.obs_kf_ids.data_ptr(),
                    self.obs_pt2d_ids.data_ptr(),
                    self.obs_uvs.data_ptr(),
                    self._obs_size_gpu.data_ptr(),
                    self.n_obs.data_ptr(),
                    desc_kpts.id_in_ba_problem.data_ptr(),
                ),
            )

        # Read back counter with a single sync
        self.obs_size = int(self._obs_size_gpu[0])

    def _add_observations_from_other_kf(
        self,
        match,
        new_landmark_mask,
        keyframes: List[Keyframe],
        kf_id_other: int,
        desc_kpts,
    ):
        """Add observations from another keyframe to newly created landmarks.

        The landmarks were just created from current keyframe's points.
        This adds the corresponding 2D observations from the matched other keyframe.
        """
        n_match = match.idx.shape[0]
        if n_match == 0:
            return

        other_desc_kpts = keyframes[kf_id_other].desc_kpts
        kpts_other = other_desc_kpts.kpts.cuda()  # ensure on GPU
        other_desc_kpts.id_in_ba_problem = other_desc_kpts.id_in_ba_problem.cuda()

        self._ensure_obs_capacity(n_match)  # conservative reserve
        self._obs_size_gpu[0] = self.obs_size

        kf_other = keyframes[kf_id_other]
        Rt_other_3x4 = kf_other.get_Rt().detach()[:3].contiguous()
        f_other = kf_other.f if kf_other.f.numel() == 1 else kf_other.f[:1].contiguous()

        n_threads = 256
        grid  = ((n_match + n_threads - 1) // n_threads,)
        block = (n_threads,)
        cupy_stream = cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
        with cupy_stream:
            self._add_obs_from_other_kernel(
                grid=grid, block=block,
                args=(
                    n_match,
                    match.idx.data_ptr(),
                    match.idx_other.data_ptr(),
                    new_landmark_mask.data_ptr(),
                    desc_kpts.id_in_ba_problem.data_ptr(),
                    self.max_obs,
                    self.n_obs.data_ptr(),
                    kpts_other.data_ptr(),
                    kf_id_other,
                    Rt_other_3x4.data_ptr(),
                    f_other.data_ptr(),
                    kf_other.centre.data_ptr(),
                    cupy.float32(self.ba_outlier_threshold),
                    self.landmarks.data_ptr(),
                    self.obs_lm_ids.data_ptr(),
                    self.obs_kf_ids.data_ptr(),
                    self.obs_pt2d_ids.data_ptr(),
                    self.obs_uvs.data_ptr(),
                    self._obs_size_gpu.data_ptr(),
                    self.n_obs.data_ptr(),
                    other_desc_kpts.id_in_ba_problem.data_ptr(),
                ),
            )

        self.obs_size = int(self._obs_size_gpu[0])

    def append(self, keyframes: List[Keyframe], keyframe: Keyframe):
        desc_kpts = keyframe.desc_kpts
        valid_mask = desc_kpts.has_pt3d_flat()
        existing_mask = torch.zeros_like(valid_mask)

        # Pose / focal for the reprojection-error gate in the kernel
        Rt_curr_3x4 = keyframe.get_Rt().detach()[:3].contiguous()
        f_curr = keyframe.f if keyframe.f.numel() == 1 else keyframe.f[:1].contiguous()

        # Update existing landmarks with observations from current keyframe
        n_matches = [len(m.kpts) for m in desc_kpts.matches.values() if not m.is_loop]
        max_matches = max(n_matches) if n_matches else 0
        for kf_id_other, match in desc_kpts.matches.items():
            self._add_observations_to_existing(
                match,
                existing_mask,
                keyframes,
                kf_id_other,
                desc_kpts,
                keyframe,
                Rt_curr_3x4,
                f_curr,
            )
            # Update the pose graph
            if not match.is_loop and (
                len(match.kpts) >= self.min_matches_edge or len(match.kpts) == max_matches):
                self.pose_graph.add_match(keyframe.index, kf_id_other, 1.0 / len(match.kpts))

        # Create new landmarks from remaining valid points
        new_landmark_mask = valid_mask & ~existing_mask
        valid_ids = new_landmark_mask.nonzero(as_tuple=False).squeeze(-1)
        n_new = valid_ids.numel()

        if n_new > 0:
            self._ensure_capacity(n_new)
            start_idx = self.size
            end_idx = start_idx + n_new

            # Add new landmarks through the flat keypoint views.
            self.landmarks    [start_idx:end_idx] = desc_kpts.pts3d_flat()[valid_ids]
            self.lm_primary_kf[start_idx:end_idx] = keyframe.index
            self.n_obs        [start_idx:end_idx] = 1

            # Link keypoints to their landmarks
            new_lm_ids = torch.arange(
                start_idx, end_idx, device="cuda", dtype=desc_kpts.id_in_ba_problem.dtype
            )
            desc_kpts.id_in_ba_problem_flat()[valid_ids] = new_lm_ids
            self.size = end_idx

            # First observation: from the current keyframe
            self._ensure_obs_capacity(n_new)
            s = self.obs_size
            self.obs_lm_ids  [s:s + n_new] = new_lm_ids.int()
            self.obs_kf_ids  [s:s + n_new] = keyframe.index
            self.obs_pt2d_ids[s:s + n_new] = valid_ids.int()
            self.obs_uvs     [s:s + n_new] = desc_kpts.kpts_flat()[valid_ids]
            self.obs_size += n_new

            # Add observations from other keyframes to newly created landmarks
            for kf_id_other, match in desc_kpts.matches.items():
                self._add_observations_from_other_kf(
                    match, new_landmark_mask, keyframes, kf_id_other, desc_kpts
                )
        keyframe.desc_kpts.ba_problem = self

    def get_landmarks(self):
        return self.landmarks[:self.size]

    def _build_dense_obs(self, selected_ids, keyframes_ids):
        """Build dense [n_sel, max_obs] observation tensors for BA.

        Only observations from keyframes_ids are included.
        Returns:
            uvs          [n_sel, max_obs, 2]  float  (-1 = empty)
            cam_ids      [n_sel, max_obs]     int32  (-1 = empty)
            kf_ids_dense [n_sel, max_obs]     int32  (-1 = empty)
        """
        n_sel = len(selected_ids)

        # lm_to_local: global landmark id -> local row index in output (-1 if not selected)
        lm_to_local = self._lm_to_local_scratch[:self.size]
        lm_to_local.fill_(-1)
        lm_to_local[selected_ids] = torch.arange(n_sel, dtype=torch.int32, device="cuda")

        max_kf_id = int(keyframes_ids.max())
        kf_selected  = torch.zeros(max_kf_id + 1, dtype=torch.int32, device="cuda")
        kf_id_to_cam = torch.full( [max_kf_id + 1], -1, dtype=torch.int32, device="cuda")
        kf_selected [keyframes_ids.long()] = 1
        kf_id_to_cam[keyframes_ids.long()] = torch.arange(len(keyframes_ids), dtype=torch.int32, device="cuda")

        # Output tensors
        uvs          = torch.full([n_sel, self.max_obs, 2], -1.0,  device="cuda")
        kf_ids_dense = torch.full([n_sel, self.max_obs],   -1, dtype=torch.int32, device="cuda")
        cam_ids      = torch.full([n_sel, self.max_obs],   -1, dtype=torch.int32, device="cuda")

        # Deterministic dense-obs build (replaces a kernel that assigned per-landmark
        # slots via a racy atomicAdd -- the surviving non-determinism after the
        # fixed-point BA kernels). On overflow (>max_obs obs for a landmark) the
        # original kept a race-decided subset; here we keep the max_obs observations
        # with the smallest cam index (earliest keyframes in the selected window), a
        # deterministic, parallax-friendly choice. Slot order within a landmark is
        # irrelevant to the (order-independent) fixed-point JtJ accumulation, so we
        # only need a deterministic *set* and rank.
        obs_lm = self.obs_lm_ids[:self.obs_size]
        obs_kf = self.obs_kf_ids[:self.obs_size]
        local = lm_to_local[obs_lm.long()]                       # -1 if landmark not selected
        keep = (local >= 0) & (kf_selected[obs_kf.long()] > 0)
        oidx = keep.nonzero(as_tuple=True)[0]                     # ascending -> deterministic
        if oidx.numel() > 0:
            loc = local[oidx].long()
            cam = kf_id_to_cam[obs_kf[oidx].long()].long()
            base = int(kf_id_to_cam.max()) + 2                   # > any cam index
            # The flat obs arrays are filled via a racy atomicAdd slot counter,
            # so their order is non-deterministic. Sorting by (landmark, cam)
            # alone leaves observations that share a (landmark, cam) key in an
            # order decided by that race -- and when a landmark has >max_obs
            # observations the `rank < max_obs` cap below would then keep a
            # race-dependent subset, flipping a few uvs and cascading into
            # pose divergence run-to-run. Break the tie deterministically with
            # the observing keypoint id (pt2d): equal (landmark, cam) entries
            # are ordered by pt2d, which is content (not append order). A
            # two-pass stable sort gives the lexicographic (landmark, cam,
            # pt2d) order without composite-key overflow concerns.
            pt2 = self.obs_pt2d_ids[oidx].long()
            order_pt = torch.argsort(pt2, stable=True)
            order = order_pt[torch.argsort((loc * base + cam)[order_pt], stable=True)]
            loc_s = loc[order]
            m = loc_s.numel()
            ar = torch.arange(m, device="cuda")
            is_new = torch.ones(m, dtype=torch.bool, device="cuda")
            is_new[1:] = loc_s[1:] != loc_s[:-1]
            # rank within landmark group = position - index of the group's first element
            group_first = torch.cummax(torch.where(is_new, ar, torch.zeros_like(ar)), 0).values
            rank = ar - group_first
            sel = rank < self.max_obs
            o = oidx[order[sel]]                                 # kept obs indices (into obs_*)
            r = rank[sel]
            l = loc_s[sel]
            uvs         [l, r] = self.obs_uvs[o]
            kf_ids_dense[l, r] = self.obs_kf_ids[o]
            cam_ids     [l, r] = kf_id_to_cam[self.obs_kf_ids[o].long()]

        return uvs, cam_ids, kf_ids_dense

    @torch.no_grad()
    def select_keyframes(self, seed_keyframes_ids: List[int], get_n_keyframes: int = -1, shuffle_ratio=0.0):
        get_n_keyframes = self.num_keyframes_localba if get_n_keyframes == -1 else get_n_keyframes
        self.pose_graph._sync_graph()
        n_keyframes = self.pose_graph.vcount()
        if n_keyframes <= get_n_keyframes:
            selected_ids = list(range(n_keyframes))
        else:            
            # Start with seed
            selected_ids = set(seed_keyframes_ids)

            while len(selected_ids) < get_n_keyframes:
                # Count connections to selected set for each candidate
                neighbor_counts = Counter()
                for sid in selected_ids:
                    for neighbor in self.pose_graph.neighbors(sid):
                        if neighbor not in selected_ids and neighbor < n_keyframes:
                            # Edge weight is 1 / n_matches, so invert to score by matches.
                            neighbor_counts[neighbor] += 1.0 / self.pose_graph.edge_weight(sid, neighbor)
                
                if not neighbor_counts:
                    break
                
                best_candidate, best_count = neighbor_counts.most_common(1)[0]
                if best_count == 0:
                    break
                selected_ids.add(best_candidate)
        
        selected_ids = sorted(list(selected_ids))
        if shuffle_ratio > 0:
            n_shuffle = int(len(selected_ids) * shuffle_ratio)
            selected_ids = random.sample(selected_ids[:n_shuffle], n_shuffle) + selected_ids[n_shuffle:]

        return selected_ids

    @torch.no_grad()
    def select_points(self, keyframes_ids: List[int], n_points: int) -> List[int]:
        ## Select a subset of landmarks for optimization
        obs_lm = self.obs_lm_ids[:self.obs_size]
        obs_kf = self.obs_kf_ids[:self.obs_size]

        # Count per-landmark observations that come from the relevant keyframes.
        # scatter_add_ with relevant.int() avoids boolean indexing + avoids 2 syncs
        # (no .any() guard, no int(relevant.sum()) for ones-tensor).
        relevant = torch.isin(obs_kf, keyframes_ids)
        n_curr_obs = torch.zeros(self.size, dtype=torch.int32, device="cuda")
        n_curr_obs.scatter_add_(0, obs_lm.long(), relevant.int())

        mask = n_curr_obs >= 2  # At least a pair of observations

        # Select points that are supervised by the relevant keyframes
        n_avail = int(mask.sum())
        if n_avail == 0:
            return torch.empty(0, dtype=torch.long, device="cuda")
        # Weight the random subsampling by observation count
        weights = n_curr_obs.float() * mask
        selected_ids = torch.multinomial(weights, min(n_avail, n_points))
        return selected_ids

    @torch.no_grad()
    def local_ba(
        self, keyframes: List[Keyframe], f: torch.Tensor, centre: torch.Tensor, keyframes_new: List[Keyframe] = [], fixed_masks: torch.Tensor = None
    ):
        """
        Select a subset of landmarks and run the optimization.
        Update the landmarks here, the keyframes' poses and their 3Dpts (for pnp)
        """
        ## Select a subset of landmarks for optimization
        keyframes_ids = torch.tensor([kf.index for kf in keyframes], dtype=int, device="cuda")
        if len(keyframes_new) > 0:
            # Select same amount of points from old and new
            keyframes_new_ids = torch.tensor([kf.index for kf in keyframes_new], dtype=int, device="cuda")
            selected_ids = self.select_points(keyframes_new_ids, self.num_pts_localba // 2)
            selected_ids = torch.cat(
                [selected_ids, self.select_points(keyframes_ids, self.num_pts_localba // 2)])
            keyframes_ids = torch.cat([keyframes_ids, keyframes_new_ids])
            keyframes = keyframes + keyframes_new
            # selected_ids, self.select_points(keyframes_ids, self.num_pts_localba)
        else:
            selected_ids = self.select_points(keyframes_ids, self.num_pts_localba)

        # Ensure not too few points
        if len(selected_ids) < len(keyframes_ids) * self.ba_min_pts_per_keyframe:
            return []

        ## Organise the BA problem
        Rts_init = torch.stack([kf.get_Rt() for kf in keyframes], dim=0)
        if len(keyframes) < self.num_keyframes_localba:
            Rts_init = torch.cat(
                [
                    Rts_init,
                    torch.eye(4, device="cuda")[None].expand(
                        self.num_keyframes_localba - len(keyframes), -1, -1
                    ).clone(),
                ]
            )

        xyz_init = self.get_landmarks()[selected_ids]
        uvs, cam_ids, kf_ids_dense = self._build_dense_obs(selected_ids, keyframes_ids)

        if len(selected_ids) < self.num_pts_localba:
            pad = self.num_pts_localba - len(selected_ids)
            xyz_init = torch.cat([xyz_init,
                torch.zeros(pad, 3, device="cuda")])
            uvs = torch.cat([uvs,
                torch.full([pad, self.max_obs, 2], -1.0, device="cuda")])
            cam_ids = torch.cat([cam_ids,
                torch.full([pad, self.max_obs], -1, dtype=cam_ids.dtype, device="cuda")])

        ## Run the optimizer
        latest_kf = keyframes_ids.max().item()
        optimizing_intrinsics = self.opt_f_until > latest_kf
        miniBA_local = self.miniBA_local_f if optimizing_intrinsics else self.miniBA_local
        if fixed_masks is None:
            fixed_masks = torch.zeros(
                miniBA_local.iters, miniBA_local.n_cams, device="cuda", dtype=torch.bool)
            n_fixed = int(self.num_keyframes_localba * self.localba_fixed_ratio)
            n_can_fix = max(int(len(keyframes) * 0.75), n_fixed)
            rand_sample = torch.rand(fixed_masks.shape[0], n_can_fix, device="cuda")
            fixed_masks[:, :n_can_fix] = rand_sample >= torch.topk(rand_sample, n_fixed, dim=1).values[:, [-1]]
        Rts, f, xyz, r, r_init, mask, eff_iters = miniBA_local(
            Rts_init,
            f,
            xyz_init,
            centre,
            uvs,
            cam_ids,
            fixed_masks=fixed_masks,
        )

        # Mask out the invalid positions based on too few supervision or outliers
        xyz_mask = mask.view(xyz.shape[0], -1).sum(-1) >= 4 # At least two 2Dpts for triangulation
        xyz_mask = xyz_mask[:len(selected_ids)]
        xyz = xyz[:len(selected_ids)]
        xyz_init = xyz_init[:len(selected_ids)]
        kf_ids_dense = kf_ids_dense[xyz_mask]
        xyz = xyz[xyz_mask]
        xyz_init = xyz_init[xyz_mask]
        selected_ids = selected_ids[xyz_mask]

        ## Assign the updated poses and 3D points accordingly
        self.landmarks[selected_ids] = xyz

        # Batch approx_centre: replaces 20 individual R.T@t matmuls with one bmm
        n_kf = len(keyframes)
        Rs_batch = Rts[:n_kf, :3, :3]
        ts_batch = Rts[:n_kf, :3, 3]
        approx_centres = torch.bmm(-Rs_batch.transpose(1, 2), ts_batch.unsqueeze(-1)).squeeze(-1)

        for i, (keyframe, Rt) in enumerate(zip(keyframes, Rts)):
            keyframe.last_opt = latest_kf
            keyframe.set_Rt(Rt)
            keyframe.approx_centre = approx_centres[i]
            if optimizing_intrinsics:
                keyframe.f.copy_(f)

        return xyz, xyz_init, selected_ids, kf_ids_dense

    def _prepare_loop_sides(self, all_keyframes, prev_keyframes_odo, prev_keyframes_loop, Rt_loop):
        prev_keyframes_odo_ids = [kf.index for kf in prev_keyframes_odo]
        prev_keyframes_loop_ids = [kf.index for kf in prev_keyframes_loop]

        ## Get the oldest set to know which one to fix
        older_set, newer_set = sorted([prev_keyframes_odo_ids, prev_keyframes_loop_ids], key=min)

        ## Select keyframes on each side of the loop
        num_kf_new = self.num_keyframes_localba // 2
        num_kf_older = num_kf_new + self.num_keyframes_localba % 2 # Add the potential extra keyframe to the older set
        opt_keyframes_old_ids = self.select_keyframes(older_set, num_kf_older)
        opt_keyframes_new_ids = self.select_keyframes(newer_set, num_kf_new)

        old_id = self.pose_graph.get_most_connected(opt_keyframes_old_ids)
        new_id = self.pose_graph.get_most_connected(opt_keyframes_new_ids)

        # Save the initial poses for all new keyframes
        init_poses = {}
        for kf_id in opt_keyframes_new_ids:
            init_Rw2c = all_keyframes[kf_id].get_R().clone()
            init_tw2c = all_keyframes[kf_id].get_t().clone()
            init_R = init_Rw2c.T
            init_t = - init_R @ init_tw2c
            init_poses[kf_id] = (init_Rw2c, init_tw2c, init_R, init_t)

        # Move all new keyframes following the transform
        # Very cheap way to handle it. Will BA always fix the approximation?
        delta_Rt = Rt_loop @ all_keyframes[-1].get_Rt().inverse()
        if prev_keyframes_loop[0].index in opt_keyframes_new_ids:
            delta_Rt = delta_Rt.inverse()
        for kf_id in opt_keyframes_new_ids:
            kf = all_keyframes[kf_id]
            new_Rt = delta_Rt @ kf.get_Rt()
            kf.set_Rt(new_Rt)

        opt_kf_ids = opt_keyframes_old_ids + opt_keyframes_new_ids
        return opt_keyframes_old_ids, opt_keyframes_new_ids, init_poses, opt_kf_ids

    def _run_loop_ba(self, all_keyframes, opt_keyframes_old_ids, opt_keyframes_new_ids, f, centre):
        all_xyz = torch.empty(0, 3, device="cuda")
        all_xyz_init = torch.empty(0, 3, device="cuda")
        all_selected_indices = torch.empty(0, device="cuda", dtype=torch.long)
        all_kf_ids_dense = torch.empty(0, self.max_obs, dtype=torch.int32, device="cuda")
        used_mask = torch.zeros(self.size, device="cuda", dtype=torch.bool)
        passes = 2
        for _ in range(passes):
            # Shuffle the fixed frames
            opt_keyframes_old_ids = random.sample(opt_keyframes_old_ids, len(opt_keyframes_old_ids))
            opt_keyframes_old = [all_keyframes[index] for index in opt_keyframes_old_ids]
            opt_keyframes_new = [all_keyframes[index] for index in opt_keyframes_new_ids]
            opt_kf_ids = opt_keyframes_old_ids + opt_keyframes_new_ids

            # Run local BA
            fixed_masks = torch.zeros(
                self.miniBA_local.iters, self.miniBA_local.n_cams, device="cuda", dtype=torch.bool)
            n_fixed = int(self.num_keyframes_localba * self.localba_fixed_ratio)
            rand_sample = torch.rand(fixed_masks.shape[0], len(opt_keyframes_old_ids), device="cuda")
            fixed_masks[:, :len(opt_keyframes_old_ids)] = rand_sample >= torch.topk(rand_sample, n_fixed, dim=1).values[:, [-1]]

            xyz, xyz_init, selected_indices, kf_ids_dense = self.local_ba(opt_keyframes_old, f, centre, opt_keyframes_new, fixed_masks)

            # Get initial and optimize points (will be used for pose scale estimation)
            already_used_mask = used_mask[selected_indices]

            # Add new points
            all_xyz = torch.cat((all_xyz, xyz[~already_used_mask]), dim=0)
            all_xyz_init = torch.cat((all_xyz_init, xyz_init[~already_used_mask]), dim=0)
            all_selected_indices = torch.cat((all_selected_indices, selected_indices[~already_used_mask]), dim=0)
            all_kf_ids_dense = torch.cat((all_kf_ids_dense, kf_ids_dense[~already_used_mask]), dim=0)
            used_mask[selected_indices] = True

        return all_xyz, all_xyz_init, all_selected_indices, all_kf_ids_dense

    def _compute_per_keyframe_transforms(self, all_keyframes, opt_keyframes_new_ids, init_poses, all_xyz, all_xyz_init, all_kf_ids_dense):
        all_scales = []
        all_dR = []
        all_dt = []

        for kf_id in opt_keyframes_new_ids:
            # Get initial and final poses
            init_Rw2c, init_tw2c, init_R, init_t = init_poses[kf_id]
            new_Rw2c = all_keyframes[kf_id].get_R()
            new_tw2c = all_keyframes[kf_id].get_t()

            # Filter points belonging to this keyframe
            relevant_mask = (all_kf_ids_dense == kf_id).sum(-1) > 0
            xyz_init = all_xyz_init[relevant_mask]
            xyz = all_xyz[relevant_mask]

            # Compute scale from inverse depth
            if len(xyz_init) > 0:
                invdepth_init_cam = 1 / (xyz_init @ init_Rw2c.T + init_tw2c[None])[..., -1]
                invdepth_cam = 1 / (xyz @ new_Rw2c.T + new_tw2c[None])[..., -1]
                mask = (invdepth_init_cam > 1e-3) * (invdepth_cam > 1e-3) * (invdepth_init_cam < 1e3) * (invdepth_cam < 1e3)
                if mask.sum() > 0:
                    scale = (invdepth_init_cam[mask] / invdepth_cam[mask]).median()
                else:
                    scale = torch.tensor(1.0, device="cuda")
            else:
                scale = torch.tensor(1.0, device="cuda")

            # Compute pose update
            new_R = new_Rw2c.T
            new_t = - new_R @ new_tw2c

            delta_R = new_R @ init_R.T

            # The equation is: new_t = s * delta_R @ init_t + delta_t
            # Therefore: delta_t = new_t - s * delta_R @ init_t
            delta_t = (new_t - scale * (delta_R @ init_t))

            all_scales.append(scale)
            all_dR.append(delta_R)
            all_dt.append(delta_t)

        # Stack into tensors
        gaussian_relocation_parameters = {
            'kf_id_tensor': torch.tensor(opt_keyframes_new_ids, device="cuda", dtype=torch.int),
            'all_scales': torch.stack(all_scales),  # (N,)
            'all_dR': torch.stack(all_dR),  # (N, 3, 3)
            'all_dt': torch.stack(all_dt)  # (N, 3)
        }

        return gaussian_relocation_parameters, all_scales, all_dR, all_dt

    def _propagate_pose_updates(self, all_keyframes, opt_keyframes_old_ids, opt_keyframes_new_ids, opt_kf_ids, gaussian_relocation_parameters, all_scales, all_dR, all_dt):
        # Get the backbone from the connectivity graph
        dists = torch.tensor(self.pose_graph.get_dists(
            opt_keyframes_old_ids, opt_keyframes_new_ids, w_weights=True))
        argmin = dists.argmin()
        old_idm = opt_keyframes_old_ids[argmin // len(opt_keyframes_new_ids)]
        new_idm = opt_keyframes_new_ids[argmin % len(opt_keyframes_new_ids)]
        backbone, weights = self.pose_graph.get_backbone_path(old_idm, new_idm)

        # Get representative transformation from new_idm for propagation
        all_dt_tensor = torch.stack(all_dt)  # (N, 3)
        median_dt = all_dt_tensor.median(dim=0).values
        new_idm_idx = (all_dt_tensor - median_dt).norm(dim=1).argmin().item()
        scale = all_scales[new_idm_idx]
        delta_R = all_dR[new_idm_idx]
        delta_t = all_dt[new_idm_idx]
        delta_R6D = mtx2sixD(delta_R)
        eye_R6D = torch.eye(3, 2, device=delta_R.device)

        # Get the backbone node each keyframe belongs to
        ownership = self.pose_graph.get_backbone_ownership_w(opt_keyframes_old_ids+backbone)

        # Propagate the pose update through the backbone (batched)
        all_kf_ids = []
        all_weights = []
        weights_sum = sum(weights)
        weight = 0
        for backbone_id, edge_weight in zip(backbone[1:], weights):
            weight += edge_weight / weights_sum
            for kf_id in ownership[backbone_id]:
                if kf_id not in opt_kf_ids:
                    all_kf_ids.append(kf_id)
                    all_weights.append(weight)

        if all_kf_ids:
            # Batch collect all poses
            all_k_R = torch.stack([all_keyframes[kf_id].get_R() for kf_id in all_kf_ids]).transpose(-1, -2)
            all_k_t = torch.stack([all_keyframes[kf_id].get_t() for kf_id in all_kf_ids])
            all_k_t = - torch.bmm(all_k_R, all_k_t.unsqueeze(-1))
            w_tensor = torch.tensor(all_weights, device="cuda", dtype=all_k_R.dtype)

            # Batch compute all dR, dt, and scale
            # Interpolate Rotation (Linear 6D)
            all_R6D = eye_R6D[None] * (1 - w_tensor[:, None, None]) + delta_R6D[None] * w_tensor[:, None, None]
            all_dR = sixD2mtx(all_R6D)

            # Interpolate Scale (Log-Linear)
            all_scales = torch.pow(scale, w_tensor)

            # Interpolate Translation (Linear global shift)
            all_dt = delta_t[None] * w_tensor[:, None]

            # Apply Update: p_new = s * dR * p_old + dt
            corrected_R = torch.bmm(all_dR, all_k_R)
            corrected_t = all_scales[:, None, None] * torch.bmm(all_dR, all_k_t) + all_dt.unsqueeze(-1)

            # Convert back to W2C
            corrected_R_W2C = corrected_R.transpose(-2, -1)
            corrected_t_W2C = -torch.bmm(corrected_R_W2C, corrected_t).squeeze(-1)

            # Write back keyframe poses
            for i, kf_id in enumerate(all_kf_ids):
                new_Rt = torch.eye(4, device="cuda")
                new_Rt[:3, :3] = corrected_R_W2C[i]
                new_Rt[:3, 3] = corrected_t_W2C[i]
                all_keyframes[kf_id].set_Rt(new_Rt)

            # Identify affected landmarks
            primary_kf_ids = self.lm_primary_kf[:self.size]
            kf_id_tensor = torch.tensor(all_kf_ids, device="cuda", dtype=torch.int) # IDs involved in propagation

            # Concatenate propagated keyframes with optimized keyframes for gaussian relocation
            new_gs_relocation_parameters = {
                'kf_id_tensor': kf_id_tensor,
                'all_scales': all_scales,
                'all_dR': all_dR,
                'all_dt': all_dt,
            }
            gaussian_relocation_parameters = {
                key: torch.cat([gaussian_relocation_parameters[key], new_gs_relocation_parameters[key]])
                for key in gaussian_relocation_parameters
            }


            # Optimization: Use bucket search or sort if N is huge, but this is fine for <50k points
            landmark_mask = torch.isin(primary_kf_ids, kf_id_tensor)

            if landmark_mask.any():
                affected_landmark_ids = torch.where(landmark_mask)[0]
                affected_primary_kfs = primary_kf_ids[affected_landmark_ids]

                # Find which index in 'all_kf_ids' corresponds to each landmark's parent
                # (N_land, 1) == (1, N_kfs)
                kf_indices = (affected_primary_kfs.unsqueeze(1) == kf_id_tensor.unsqueeze(0)).int().argmax(dim=1)

                #Gather the Sim3 params applied to the parents
                # You need to have kept these from the pose section:
                # all_scales: (N_kfs, )
                # all_dR:     (N_kfs, 3, 3)
                # all_dt:     (N_kfs, 3)

                l_scales = all_scales[kf_indices]       # (N_affected, )
                l_dR = all_dR[kf_indices]               # (N_affected, 3, 3)
                l_dt = all_dt[kf_indices]               # (N_affected, 3)

                # Apply the Sim3 Transform
                # P_new = s * R * P_old + t

                # Get old points
                old_points = self.landmarks[affected_landmark_ids] # (N_affected, 3)

                # Rotate
                # (N, 3, 3) @ (N, 3, 1) -> (N, 3, 1)
                rotated_points = torch.bmm(l_dR, old_points.unsqueeze(-1)).squeeze(-1)

                # Scale and Translate
                # Note: l_scales needs reshape for broadcast (N, 1)
                new_points = l_scales.unsqueeze(1) * rotated_points + l_dt

                # Update in place
                self.landmarks[affected_landmark_ids] = new_points

        return gaussian_relocation_parameters

    def close_loop(self, all_keyframes: List[Keyframe], prev_keyframes_odo: List[Keyframe], prev_keyframes_loop: List[Keyframe], Rt_loop: torch.Tensor, f: torch.Tensor, centre: torch.Tensor):
        opt_keyframes_old_ids, opt_keyframes_new_ids, init_poses, opt_kf_ids = self._prepare_loop_sides(
            all_keyframes, prev_keyframes_odo, prev_keyframes_loop, Rt_loop)

        all_xyz, all_xyz_init, all_selected_indices, all_kf_ids_dense = self._run_loop_ba(
            all_keyframes, opt_keyframes_old_ids, opt_keyframes_new_ids, f, centre)

        gaussian_relocation_parameters, all_scales, all_dR, all_dt = self._compute_per_keyframe_transforms(
            all_keyframes, opt_keyframes_new_ids, init_poses, all_xyz, all_xyz_init, all_kf_ids_dense)

        gaussian_relocation_parameters = self._propagate_pose_updates(
            all_keyframes, opt_keyframes_old_ids, opt_keyframes_new_ids, opt_kf_ids,
            gaussian_relocation_parameters, all_scales, all_dR, all_dt)

        ## Validate the loop edges and add them to the pose graph
        for index in opt_kf_ids:
            keyframe = all_keyframes[index]
            for kf_id_other, match in keyframe.desc_kpts.matches.items():
                if match.is_loop and kf_id_other in opt_kf_ids:
                    match.is_loop = False
                    if len(match.kpts) >= self.min_matches_edge:
                        self.pose_graph.add_match(
                            keyframe.index, kf_id_other, 1.0 / len(match.kpts))

        return gaussian_relocation_parameters

    def global_ba(
        self, keyframes: List[Keyframe], f: torch.Tensor, centre: torch.Tensor
    ):
        keyframes_ids = torch.tensor([kf.index for kf in keyframes], dtype=int, device="cuda")
        num_pts = len(keyframes) * self.global_ba_pts_per_keyframe
        selected_ids = self.select_points(keyframes_ids, num_pts)

        # Ensure not too few points
        if len(selected_ids) < len(keyframes_ids) * self.ba_min_pts_per_keyframe:
            return []

        ## Organise the BA problem
        Rts_init = torch.stack([kf.get_Rt() for kf in keyframes], dim=0)

        xyz_init = self.get_landmarks()[selected_ids]
        uvs, cam_ids, kf_ids_dense = self._build_dense_obs(selected_ids, keyframes_ids)

        # Fix cameras that have too few observations — under-constrained poses must not be updated
        n_cams = len(keyframes)
        obs_per_cam = torch.zeros(n_cams, dtype=torch.int32, device="cuda")
        valid_obs = cam_ids[cam_ids >= 0]
        obs_per_cam.scatter_add_(0, valid_obs.long(), torch.ones_like(valid_obs))
        under_constrained = obs_per_cam < self.ba_min_pts_per_keyframe

        ## Run the optimizer
        latest_kf = keyframes_ids.max().item()
        miniBA = MiniBA(
            n_cams,
            num_pts,
            self.max_obs,
            optimize_focal=False,
            optimize_3Dpts=True,
            make_cuda_graph=False,
            outlier_threshold=self.ba_outlier_threshold,
            iters=10,
            lm=1e-4,
            outlier_mad_scale=8.0,
        )
        fixed_masks = under_constrained.unsqueeze(0).expand(miniBA.iters, -1)
        Rts, f, xyz, r, r_init, mask, eff_iters = miniBA(
            Rts_init,
            f,
            xyz_init,
            centre,
            uvs,
            cam_ids,
            fixed_masks=fixed_masks,
        )
        # Mask out the invalid positions based on too few supervision or outliers
        xyz_mask = mask.view(xyz.shape[0], -1).sum(-1) >= 4 # At least two 2Dpts for triangulation
        xyz_mask = xyz_mask[:len(selected_ids)]
        xyz = xyz[:len(selected_ids)]
        xyz_init = xyz_init[:len(selected_ids)]
        xyz = xyz[xyz_mask]
        xyz_init = xyz_init[xyz_mask]
        selected_ids = selected_ids[xyz_mask]

        final_residual = (r * mask).abs().sum() / mask.sum()
        initial_residual = (r_init * mask).abs().sum() / mask.sum()
        print(
            f"f: {f.item():.2f}, initial residual: {initial_residual.item():.3f}, final residual: {final_residual.item():.3f}, eff_iters: {eff_iters.item()}"
        )

        ## Assign the updated poses and 3D points accordingly
        self.landmarks[selected_ids] = xyz
        for i, (keyframe, Rt) in enumerate(zip(keyframes, Rts)):
            if under_constrained[i]:
                continue
            keyframe.last_opt = latest_kf
            keyframe.set_Rt(Rt)
            if self.opt_f_until > latest_kf:
                keyframe.f.copy_(f)

        return xyz, xyz_init, selected_ids
