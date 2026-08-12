import torch
from torch import nn


class ContinuousScaffoldField(nn.Module):
    """
    Minimal 3D LIIF-style continuous scaffold field.

    This first version is query-only:
    - sparse anchors stay unchanged
    - for any 3D query point, gather K nearest anchors
    - use local ensemble + cell-conditioned decoding
    - predict a residual RGB on top of the stage-1 Scaffold-GS render
    """

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 64,
        k_neighbors: int = 8,
        temperature: float = 0.05,
        knn_chunk_size: int = 8192,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.temperature = temperature
        self.knn_chunk_size = knn_chunk_size

        local_in_dim = feat_dim + 3 + 3 + 3 + 1
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
            nn.Linear(hidden_dim + 3 + 3, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, feat_dim),
        )

        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def _knn(self, query_xyz: torch.Tensor, anchor_xyz: torch.Tensor):
        k = min(self.k_neighbors, anchor_xyz.shape[0])
        dist_chunks = []
        idx_chunks = []
        for start in range(0, query_xyz.shape[0], self.knn_chunk_size):
            end = min(start + self.knn_chunk_size, query_xyz.shape[0])
            dists = torch.cdist(query_xyz[start:end], anchor_xyz)
            knn_dist, knn_idx = torch.topk(
                dists,
                k=k,
                dim=1,
                largest=False,
                sorted=True,
            )
            dist_chunks.append(knn_dist)
            idx_chunks.append(knn_idx)
        return torch.cat(dist_chunks, dim=0), torch.cat(idx_chunks, dim=0)

    def forward(
        self,
        query_xyz: torch.Tensor,
        query_cell: torch.Tensor,
        query_viewdir: torch.Tensor,
        anchor_xyz: torch.Tensor,
        anchor_feat: torch.Tensor,
    ):
        knn_dist, knn_idx = self._knn(query_xyz, anchor_xyz)

        neigh_xyz = anchor_xyz[knn_idx]
        neigh_feat = anchor_feat[knn_idx]

        rel_xyz = query_xyz[:, None, :] - neigh_xyz
        cell = query_cell[:, None, :].expand_as(rel_xyz)
        viewdir = query_viewdir[:, None, :].expand_as(rel_xyz)
        dist_feat = knn_dist.unsqueeze(-1)

        local_input = torch.cat([neigh_feat, rel_xyz, cell, viewdir, dist_feat], dim=-1)
        local_feat = self.local_mlp(local_input)

        weights = torch.softmax(-knn_dist / self.temperature, dim=1).unsqueeze(-1)
        fused = (weights * local_feat).sum(dim=1)
        fused = self.fuse_mlp(fused)

        dense_feat = self.feature_head(torch.cat([fused, query_viewdir, query_cell], dim=-1))
        confidence = self.conf_head(fused)
        return dense_feat, confidence, knn_idx, knn_dist
