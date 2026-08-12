import torch
from torch import nn


class LIIFAnchorField(nn.Module):
    """
    3D LIIF-style field for anchor-conditioned continuous feature queries.

    Query feature at anchor position x using:
    - local ensemble over K nearest frozen stage-1 anchors
    - feature unfolding via neighbor feature gathering
    - cell decoding via local spatial cell size
    """

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 64,
        k_neighbors: int = 8,
        temperature: float = 0.05,
        knn_chunk_size: int = 1024,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.temperature = temperature
        self.knn_chunk_size = knn_chunk_size

        local_in_dim = feat_dim + 3 + 3 + 1
        self.local_mlp = nn.Sequential(
            nn.Linear(local_in_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.fuse_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.feature_head = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, feat_dim),
        )
        self._init_zero_output()

    def _init_zero_output(self):
        # Stage-2 should start as an identity refinement on top of stage-1.
        # Zero-initializing the last projection guarantees the LIIF branch
        # contributes exactly zero at initialization.
        last = self.feature_head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def _knn(self, query_xyz: torch.Tensor, anchor_xyz: torch.Tensor, k_override: int = None):
        k_target = self.k_neighbors if k_override is None else int(k_override)
        k = min(k_target, anchor_xyz.shape[0])
        dist_chunks = []
        idx_chunks = []
        for start in range(0, query_xyz.shape[0], self.knn_chunk_size):
            end = min(start + self.knn_chunk_size, query_xyz.shape[0])
            dists = torch.cdist(query_xyz[start:end], anchor_xyz)
            knn_dist, knn_idx = torch.topk(dists, k=k, dim=1, largest=False, sorted=True)
            dist_chunks.append(knn_dist)
            idx_chunks.append(knn_idx)
        return torch.cat(dist_chunks, dim=0), torch.cat(idx_chunks, dim=0)

    def forward(
        self,
        query_xyz: torch.Tensor,
        query_cell: torch.Tensor,
        context_xyz: torch.Tensor,
        context_feat: torch.Tensor,
        k_override: int = None,
    ):
        knn_dist, knn_idx = self._knn(query_xyz, context_xyz, k_override=k_override)
        neigh_xyz = context_xyz[knn_idx]
        neigh_feat = context_feat[knn_idx]
        rel_xyz = query_xyz[:, None, :] - neigh_xyz
        cell = query_cell[:, None, :].expand_as(rel_xyz)
        dist_feat = knn_dist.unsqueeze(-1)

        local_input = torch.cat([neigh_feat, rel_xyz, cell, dist_feat], dim=-1)
        local_feat = self.local_mlp(local_input)
        weights = torch.softmax(-knn_dist / self.temperature, dim=1).unsqueeze(-1)
        fused = (weights * local_feat).sum(dim=1)
        fused = self.fuse_mlp(fused)
        return self.feature_head(torch.cat([fused, query_cell], dim=-1))
