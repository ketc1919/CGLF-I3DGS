# LighterGlue implementation adapted from:
# - https://github.com/verlab/accelerated_features (Apache-2.0 license)
# - Kornia LightGlue (Apache-2.0 license)
# A copy of the Apache License 2.0 is provided in licenses/Apache-2.0.txt.
# Local copy to allow optimizations and avoid external dependencies

import math
import warnings
from types import SimpleNamespace
from typing import Optional, Tuple, Dict
import torch
import torch.nn.functional as F
from torch import nn
import os

try:
    from flash_attn.modules.mha import FlashCrossAttention
except ModuleNotFoundError:
    FlashCrossAttention = None

if FlashCrossAttention or hasattr(F, "scaled_dot_product_attention"):
    FLASH_AVAILABLE = True
else:
    FLASH_AVAILABLE = False


def normalize_keypoints(kpts: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """Normalize tensor of keypoints."""
    if isinstance(size, torch.Size):
        size = torch.tensor(size)[None]
    shift = size.float().to(kpts) / 2
    scale = size.max(1).values.float().to(kpts) / 2
    kpts = (kpts - shift[:, None]) / scale[:, None, None]
    return kpts


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Apply half rotation."""
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding."""
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: Optional[int] = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode position vector."""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get confidence tokens."""
        dtype = self.token[0].weight.dtype
        orig_dtype = desc0.dtype
        return (
            self.token(desc0.detach().to(dtype)).squeeze(-1).to(orig_dtype),
            self.token(desc1.detach().to(dtype)).squeeze(-1).to(orig_dtype),
        )


class Attention(nn.Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE
        self.has_sdp = hasattr(F, "scaled_dot_product_attention")
        if allow_flash and FlashCrossAttention:
            self.flash_ = FlashCrossAttention()
        if self.has_sdp:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.enable_flash and q.device.type == "cuda":
            if self.has_sdp:
                orig_dtype = q.dtype
                args = [x.half().contiguous() for x in [q, k, v]]
                v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(orig_dtype)
                return v if mask is None else v.nan_to_num()
            else:
                assert mask is None
                q, k, v = (x.transpose(-2, -3).contiguous() for x in [q, k, v])
                m = self.flash_(q.half(), torch.stack([k, v], 2).half())
                return m.transpose(-2, -3).to(q.dtype).clone()
        elif self.has_sdp:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
            attn = F.softmax(sim, -1)
            return torch.einsum("...ij,...jd->...id", attn, v)


class SelfBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, encoding: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        qk0, qk1 = self.to_qk(x0), self.to_qk(x1)
        v0, v1 = self.to_v(x0), self.to_v(x1)
        qk0, qk1, v0, v1 = (t.unflatten(-1, (self.heads, -1)).transpose(1, 2) for t in (qk0, qk1, v0, v1))
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            m1 = self.flash(qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None)
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = (t.transpose(1, 2).flatten(start_dim=-2) for t in [m0, m1])
        m0, m1 = self.to_out(m0), self.to_out(m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        encoding0: torch.Tensor,
        encoding1: torch.Tensor,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, mask0, mask1)
        else:
            desc0 = self.self_attn(desc0, encoding0)
            desc1 = self.self_attn(desc1, encoding1)
            return self.cross_attn(desc0, desc1)

    def masked_forward(
        self, desc0: torch.Tensor, desc1: torch.Tensor, encoding0: torch.Tensor,
        encoding1: torch.Tensor, mask0: torch.Tensor, mask1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        desc0 = self.self_attn(desc0, encoding0, mask0)
        desc1 = self.self_attn(desc1, encoding1, mask1)
        return self.cross_attn(desc0, desc1, mask)


def sigmoid_log_double_softmax(sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    """Create the log assignment matrix from logits and similarity."""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    
    scores = torch.full((b, m + 1, n + 1), 0, dtype=sim.dtype, device=sim.device)
    
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores

class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build assignment matrix from descriptors."""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores: torch.Tensor, th: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Obtain matches from a log assignment matrix [Bx M+1 x N+1]."""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    
    # torch.arange is safe here because sizes are fixed during graph capture
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    
    max0_exp = max0.values.exp()
    
    zero = torch.zeros_like(max0_exp)

    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    
    # DENSE OUTPUT for Graphs: -1 indicates invalid
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    
    return m0, m1, mscores0, mscores1

class LightGlue(nn.Module):
    """LightGlue feature matcher."""

    default_conf_xfeat = {
        "name": "xfeat",
        "input_dim": 64,
        "descriptor_dim": 96,
        "add_scale_ori": False,
        "add_laf": False,
        "scale_coef": 1.0,
        "n_layers": 6,
        "num_heads": 1,
        "flash": True,
        "mp": False,
        "depth_confidence": -1,
        "width_confidence": 0.95,
        "filter_threshold": 0.1,
        "weights": None,
    }

    def __init__(self, weights_path=None, use_half=True, n_layers=None):
        super().__init__()
        self.conf = SimpleNamespace(**self.default_conf_xfeat)
        if n_layers is not None:
            self.conf.n_layers = n_layers
        conf = self.conf
        self.use_half = use_half

        # Input projection
        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        # Positional encoding
        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(2, head_dim, head_dim)

        # Transformer layers
        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim
        self.transformers = nn.ModuleList([TransformerLayer(d, h, conf.flash) for _ in range(n)])
        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList([TokenConfidence(d) for _ in range(n - 1)])

        self.register_buffer(
            "confidence_thresholds",
            torch.Tensor([self.confidence_threshold(i) for i in range(conf.n_layers)]),
        )

        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load weights
        if weights_path is None:
            weights_path = os.path.expanduser('~/.cache/torch/hub/verlab_accelerated_features_main/weights/xfeat-lighterglue.pt')

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.dev)
        else:
            print("Downloading LighterGlue weights...")
            state_dict = torch.hub.load_state_dict_from_url(
                "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat-lighterglue.pt",
                map_location=self.dev
            )

        # Rename old state dict entries for compatibility
        for i in range(conf.n_layers):
            pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            state_dict = {k.replace('matcher.', ''): v for k, v in state_dict.items()}

        self.load_state_dict(state_dict, strict=False)

        # Drop unused modules — token_confidence and all but the active last
        # log_assignment/transformer are never called, so they only waste memory.
        n = conf.n_layers
        del self.token_confidence
        self.transformers  = nn.ModuleList(list(self.transformers)[:n])
        self.log_assignment = nn.ModuleList([self.log_assignment[n - 1]])

        self.to(self.dev)

        # Convert to half precision for faster inference
        if use_half and self.dev.type == 'cuda':
            self.half()

        self.eval()

    def confidence_threshold(self, layer_index: int) -> float:
        """Scaled confidence threshold."""
        threshold = 0.8 + 0.1 * math.exp(-4.0 * layer_index / self.conf.n_layers)
        return min(max(threshold, 0), 1)

    def forward(self, kpts0, kpts1, desc0, desc1, size0, size1, min_conf_tensor):
        """
        Graph-friendly forward pass.
        Accepts unpacked tensors. Returns dense tensors (indices with -1 for non-matches).

        Args:
            min_conf_tensor: MUST be a tensor (not scalar) for CUDA graph compatibility
        """
        kpts0 = normalize_keypoints(kpts0, size0)
        kpts1 = normalize_keypoints(kpts1, size1)

        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)

        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        for i in range(self.conf.n_layers):
            desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1)

        # Final Matching - extract scalar from tensor without .item()
        # This allows the value to be updated via copy_/fill_ in the static buffer
        min_conf = min_conf_tensor if isinstance(min_conf_tensor, float) else min_conf_tensor[0]
        scores, _ = self.log_assignment[0](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(scores, min_conf)

        return m0, m1, mscores0, mscores1


class CUDAGraphLighterGlue(nn.Module):
    """
    Self-contained wrapper that builds the LightGlue model internally
    and wraps it with CUDA Graphs.
    """
    @torch.no_grad()
    def __init__(self,
                 num_keypoints: int,
                 weights_path: str = None,
                 batch_size: int = 1,
                 use_half: bool = True,
                 descriptor_dim: int = 64,
                 n_layers: int = None):
        super().__init__()

        self.use_half = use_half
        self.batch_size = batch_size
        self.num_keypoints = num_keypoints
        self.descriptor_dim = descriptor_dim

        dtype_float = torch.float16 if use_half else torch.float32

        sample_kpts0 = torch.rand((batch_size, num_keypoints, 2), dtype=dtype_float, device="cuda")
        sample_kpts1 = torch.rand((batch_size, num_keypoints, 2), dtype=dtype_float, device="cuda")
        sample_desc0 = torch.randn((batch_size, num_keypoints, descriptor_dim), dtype=dtype_float, device="cuda")
        sample_desc1 = torch.randn((batch_size, num_keypoints, descriptor_dim), dtype=dtype_float, device="cuda")
        sample_size0 = torch.tensor([[1024, 1024]] * batch_size, dtype=torch.int64, device="cuda")
        sample_size1 = torch.tensor([[1024, 1024]] * batch_size, dtype=torch.int64, device="cuda")
        sample_conf = torch.tensor([0.1], dtype=torch.float32, device="cuda")

        cache_n_layers = n_layers if n_layers is not None else LightGlue.default_conf_xfeat["n_layers"]
        cache_dtype = "fp16" if use_half else "fp32"
        cache_path = os.path.join(
            "models",
            "cache",
            f"lighterglue_{cache_n_layers}l_{descriptor_dim}d_{cache_dtype}.pt",
        )

        if os.path.exists(cache_path):
            self.impl = torch.jit.load(cache_path, map_location="cuda")
        else:
            self.impl = LightGlue(weights_path, use_half=use_half, n_layers=n_layers)
            try:
                self.impl = torch.jit.trace(self.impl, (sample_kpts0, sample_kpts1,
                    sample_desc0, sample_desc1,
                    sample_size0, sample_size1,
                    sample_conf))
                self.impl = torch.jit.script(self.impl)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.jit.save(self.impl, cache_path)
                self.impl = torch.jit.load(cache_path, map_location="cuda")
            except Exception as e:
                # torch.jit.trace seems to fail for some torch/CUDA combinations
                pass

        if not torch.version.hip:
            self.impl = torch.cuda.make_graphed_callables(
                self.impl,
                (sample_kpts0, sample_kpts1,
                 sample_desc0, sample_desc1,
                 sample_size0, sample_size1,
                 sample_conf)
            )

    @torch.inference_mode()
    def __call__(
        self,
        desc_kpts: 'DescribedKeypoints',
        desc_kpts_other: 'DescribedKeypoints',
        size: torch.tensor,
        rotation_idx: int = 0,
        min_conf: float = 0.1
    ):
        """
        Matches keypoints using LighterGlue.
        Returns indices similar to match() function in matcher.py.

        Args:
            desc_kpts (DescribedKeypoints): Keypoints and descriptors of the first image.
            desc_kpts_other (DescribedKeypoints): Keypoints and descriptors of the second image.
            rotation_idx (int): Rotation index (0-3) to use for desc_kpts features.
            min_conf (float): Minimum confidence threshold for matches.

        Returns:
            tuple: (idx0, idx_other) - indices of matched keypoints in each set
        """
        # Flatten the batched keypoint pool to a single [B*N, ...] view.
        kpts0 = desc_kpts.kpts_flat()[None].clone()  # [1, N, 2]
        kpts1 = desc_kpts_other.kpts_flat()[None]    # [1, M, 2]
        desc0 = desc_kpts.full_feats_flat()[rotation_idx][None]  # [1, N, D]
        desc1 = desc_kpts_other.feats_flat()[None]   # [1, M, D]
        size0 = size

        # Transform keypoints to rotated coordinate space
        if rotation_idx == 1:  # 90° CCW: (x, y) -> (y, W-1-x)
            W = size[0, 0]
            kpts0 = torch.stack([kpts0[..., 1], W - 1 - kpts0[..., 0]], dim=-1)
            size0 = size.flip(-1)  # swap W, H
        elif rotation_idx == 2:  # 180°: (x, y) -> (W-1-x, H-1-y)
            W, H = size[0, 0], size[0, 1]
            kpts0 = torch.stack([W - 1 - kpts0[..., 0], H - 1 - kpts0[..., 1]], dim=-1)
        elif rotation_idx == 3:  # 270° CCW: (x, y) -> (H-1-y, x)
            H = size[0, 1]
            kpts0 = torch.stack([H - 1 - kpts0[..., 1], kpts0[..., 0]], dim=-1)
            size0 = size.flip(-1)  # swap W, H

        if self.use_half:
            if kpts0.dtype != torch.float16: kpts0 = kpts0.half()
            if kpts1.dtype != torch.float16: kpts1 = kpts1.half()
            if desc0.dtype != torch.float16: desc0 = desc0.half()
            if desc1.dtype != torch.float16: desc1 = desc1.half()

        conf_tensor = torch.tensor([min_conf], dtype=torch.float32, device=kpts0.device)

        m0_dense, m1_dense, mscores0, mscores1 = self.impl(
            kpts0, kpts1,
            desc0, desc1,
            size0, size,
            conf_tensor
        )

        valid_mask = m0_dense[0] > -1
        idx0 = torch.nonzero(valid_mask, as_tuple=True)[0]
        idx_other = m0_dense[0][valid_mask]

        return idx0, idx_other
