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

from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
import torch

from poses.ransac import EstimatorType, RANSACEstimator
from poses.lighterglue_matcher import CUDAGraphLighterGlue

if TYPE_CHECKING:
    from poses.feature_detector import DescribedKeypoints


class Matches:
    """
    A class to store matched keypoints and their indices between two sets of keypoints.
    """

    def __init__(self, kpts, kpts_other, idx, idx_other, is_loop=False):
        self.kpts = kpts
        self.kpts_other = kpts_other
        self.idx = idx.contiguous()
        self.idx_other = idx_other.contiguous()
        self.is_loop = is_loop


# Adapted from https://github.com/verlab/accelerated_features
def match(feats1, feats2, min_cossim=0.82):
    cossim = feats1 @ feats2.t()

    bestcossim, match12 = cossim.max(dim=1)
    _, match21 = cossim.max(dim=0)

    idx0 = torch.arange(match12.shape[0], device=match12.device)
    mask = match21[match12] == idx0

    if min_cossim > 0:
        mask *= bestcossim > min_cossim

    return idx0, match12, mask


class Matcher:
    @torch.no_grad()
    def __init__(
        self, fundmat_samples: int, max_error: float, n_kpts: int, size: Tuple[int, int],
        lg_n_layers: int = None, use_rotated_descriptors: bool = True,
        min_cossim: float = 0.82, max_candidates: int = 20,
        enable_lightglue: bool = True,
        batched_match_subset_size: int = 1024,
    ):
        """
        Initialize the Matcher.
        Args:
            fundmat_samples (int): Number of RANSAC etimations when estimating inliers with fundamental matrix estimation.
            max_error (float): Maximum error for RANSAC inlier threshold.
            n_kpts (int): Pool size of keypoints per keyframe.
            size (Tuple[int, int]): Image size (width, height) needed for LighterGlue matcher.
            lg_n_layers (int): Number of LightGlue transformer layers (default: 6).
            use_rotated_descriptors (bool): Whether rotated descriptors (4 rotations) are used.
            min_cossim (float): Minimum cosine similarity for mutual nearest neighbor matching.
            max_candidates (int): Maximum number of candidate keyframes for batched match evaluation.
            batched_match_subset_size (int): Random keypoint subset size used by
                batched_evaluate_match to keep the temporary [candidates, N, M]
                similarity tensor bounded.
        """
        self.max_error = max_error
        self.min_cossim = min_cossim
        self.disable_fundmat_filter = False
        self.fundmat_estimator = RANSACEstimator(
            fundmat_samples, max_error, EstimatorType.FUNDAMENTAL_8PTS
        )
        self.size = torch.tensor(size, device="cuda")[None]
        self.lighterglue = (
            CUDAGraphLighterGlue(n_kpts, n_layers=lg_n_layers)
            if enable_lightglue else None
        )

        n_rotations = 4 if use_rotated_descriptors else 1
        self.batched_match_subset_size = min(n_kpts, batched_match_subset_size)
        self.candidate_feats = torch.zeros(max_candidates, self.batched_match_subset_size, 64, device="cuda", dtype=torch.float16)
        self.query_feats = torch.zeros(n_rotations, self.batched_match_subset_size, 64, device="cuda", dtype=torch.float16)
        dummy_query_feats = torch.rand_like(self.query_feats)
        dummy_rotations = torch.randint(0, n_rotations, (len(self.candidate_feats),), device="cuda")
        if not torch.version.hip:
            self.batched_evaluate_match_graph = torch.cuda.make_graphed_callables(
                self.batched_evaluate_match_internal,
                (self.candidate_feats, dummy_query_feats, dummy_rotations),
            )
        else:
            self.batched_evaluate_match_graph = self.batched_evaluate_match_internal

    def batched_evaluate_match_internal(
            self, candidate_feats, query_feats, candidate_rotations):
        """
        query_feats: [4, M, D] tensor (current frame features, all 4 rotations)
        candidate_rotations: tensor of best rotation index per candidate
        Returns: tensor of match counts
        """

        # Select the correct rotation for each candidate
        selected_query_feats = query_feats[candidate_rotations]  # [n_candidates, M, D]

        # Batched cosine similarity: [n_candidates, N, M]
        cossim = torch.bmm(candidate_feats, selected_query_feats.transpose(-1, -2))

        # Best match per candidate keypoint (N → M) and per query keypoint (M → N)
        bestcossim, match12 = cossim.max(dim=2)  # [n_candidates, N]
        _, match21 = cossim.max(dim=1)  # [n_candidates, M]

        # Reciprocal check: candidate kpt i must also be the best match for its matched query kpt
        N = candidate_feats.shape[1]
        idx0 = torch.arange(N, device=cossim.device)[None].expand(candidate_feats.shape[0], -1)
        reciprocal_mask = match21.gather(1, match12) == idx0

        mask = reciprocal_mask & (bestcossim > self.min_cossim)

        return mask.sum(dim=1)  # [n_candidates]

    def batched_evaluate_match(
            self, candidate_feats_list, query_feats, candidate_rotations, subset_size: int | None = None):
        """
        candidate_feats_list: list of tensors (one per candidate keyframe, rotation 0 only)
        query_feats: [4, M, D] tensor (current frame features, all 4 rotations)
        candidate_rotations: tensor of best rotation index per candidate
        subset_size: random keypoint subset size. Defaults to the
            constructor's batched_match_subset_size, usually 1024.
        Returns: tensor of match counts
        """
        candidate_rotations = candidate_rotations.to(device=query_feats.device, dtype=torch.long)
        if query_feats.shape[0] == 1:
            candidate_rotations = torch.zeros_like(candidate_rotations)

        subset_size = self.batched_match_subset_size if subset_size is None else subset_size
        subset_size = min(subset_size, self.batched_match_subset_size)
        self.candidate_feats.zero_()
        self.query_feats.zero_()

        n_query = min(subset_size, query_feats.shape[1])
        query_subset = torch.randperm(query_feats.shape[1], device=query_feats.device)[:n_query]
        self.query_feats[:, :n_query].copy_(query_feats[:, query_subset])
        for i, feat in enumerate(candidate_feats_list):
            n_candidate = min(subset_size, feat.shape[0])
            candidate_subset = torch.randperm(feat.shape[0], device=feat.device)[:n_candidate]
            self.candidate_feats[i, :n_candidate].copy_(feat[candidate_subset])

        if len(candidate_rotations) < len(self.candidate_feats):
            candidate_rotations = torch.cat([
                candidate_rotations, 
                torch.zeros(len(self.candidate_feats) - len(candidate_rotations), dtype=torch.long, device=query_feats.device)
            ])

        scores = self.batched_evaluate_match_graph(
                self.candidate_feats, self.query_feats, candidate_rotations
            )[:len(candidate_feats_list)]
        if subset_size == self.batched_match_subset_size:
            return scores
        return scores * self.batched_match_subset_size // max(subset_size, 1)

    @torch.no_grad()
    def __call__(
        self,
        desc_kpts: DescribedKeypoints,
        desc_kpts_other: DescribedKeypoints,
        rotation_idx,
        remove_outliers: bool = False,
        update_kpts_flag: str = "",
        kID: int = -1,
        kID_other: int = -1,
        use_lightglue: bool = False,
        min_match_count: int = 0,
        loop: bool = False,
    ):
        """
        Matches keypoints between two sets of described keypoints, with optional outlier removal based on the fundamental RANSAC estimation.
        Args:
            desc_kpts (DescribedKeypoints): Keypoints and descriptors of the first image.
            desc_kpts_other (DescribedKeypoints): Keypoints and descriptors of the second image.
            remove_outliers (bool): Whether to remove outliers using the fundamental matrix.
            update_kpts_flag (str): If "all", updates all matches; if "inliers", updates only inliers.
            kID (int): ID of the first set of keypoints, used for updating matches.
            kID_other (int): ID of the second set of keypoints, used for updating matches.
            use_lightglue (bool): If True, uses LighterGlue matcher instead of mutual NN.
        Returns:
            Matches: A Matches object containing the matched keypoints and their indices.
        """
        if use_lightglue:
            if self.lighterglue is None:
                raise RuntimeError("LightGlue matching was requested, but Matcher was initialized with enable_lightglue=False.")
            # Use LighterGlue matcher
            idx, idx_other = self.lighterglue(desc_kpts, desc_kpts_other, self.size, rotation_idx)
        else:
            # Use mutual nearest neighbor matcher over the flattened keypoint pool.
            idx, idx_other, mask = match(
                desc_kpts.full_feats_flat()[rotation_idx].cuda(), desc_kpts_other.feats_flat().cuda(),
                min_cossim=self.min_cossim,
            )
            idx = idx[mask]
            idx_other = idx_other[mask]
        idx, idx_other = idx.int(), idx_other.int()
        kpts = desc_kpts.kpts_flat()[idx]
        kpts_other = desc_kpts_other.kpts_flat().cuda()[idx_other]
        idx_all = idx
        idx_other_all = idx_other
        kpts_all = kpts
        kpts_other_all = kpts_other

        if remove_outliers and not self.disable_fundmat_filter and len(kpts) >= 8:
            F, mask = self.fundmat_estimator(kpts, kpts_other)
            idx = idx[mask]
            idx_other = idx_other[mask]
            kpts = kpts[mask]
            kpts_other = kpts_other[mask]

        if len(idx) >= min_match_count:
            if update_kpts_flag == "all":
                assert kID >= 0 and kID_other >= 0
                desc_kpts.update_matches(
                    kID_other, Matches(kpts_all, kpts_other_all, idx_all, idx_other_all, loop)
                )
                desc_kpts_other.update_matches(
                    kID, Matches(kpts_other_all, kpts_all, idx_other_all, idx_all, loop)
                )
            elif update_kpts_flag == "inliers":
                assert kID >= 0 and kID_other >= 0
                desc_kpts.update_matches(
                    kID_other, Matches(kpts, kpts_other, idx, idx_other, loop)
                )
                desc_kpts_other.update_matches(
                    kID, Matches(kpts_other, kpts, idx_other, idx, loop)
                )

            return Matches(kpts, kpts_other, idx, idx_other, loop)
        else:
            return None
