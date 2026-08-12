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

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


_ARITH_TO_INT = {
    "aten::floor_divide": "aten::floordiv",
    "aten::mul": "aten::mul",
    "aten::sub": "aten::sub",
    "aten::add": "aten::add",
}


def _walk_jit_nodes(block):
    for node in block.nodes():
        yield node
        for sub in node.blocks():
            yield from _walk_jit_nodes(sub)


def _fold_shape_syncs(graph) -> int:
    int_type = torch._C.IntType.get()

    def as_int_value(value, insert_before):
        if str(value.type()) == "int":
            return value
        node = value.node()
        kind = node.kind()
        if kind == "prim::NumToTensor":
            inner = next(node.inputs())
            return inner if str(inner.type()) == "int" else None
        if kind == "prim::Constant":
            try:
                scalar = int(node.t("value").item())
            except Exception:
                return None
            const = graph.create("prim::Constant")
            const.output().setType(int_type)
            const.i_("value", scalar)
            const.insertBefore(insert_before)
            return const.output()
        if kind in _ARITH_TO_INT:
            ins = list(node.inputs())
            lhs = as_int_value(ins[0], insert_before)
            rhs = as_int_value(ins[1], insert_before)
            if lhs is None or rhs is None:
                return None
            new = graph.create(_ARITH_TO_INT[kind], [lhs, rhs])
            new.output().setType(int_type)
            new.insertBefore(insert_before)
            return new.output()
        return None

    conversion_kinds = ("aten::Int", "aten::ScalarImplicit")
    folded = 0
    for _ in range(16):
        changed = 0
        for node in list(_walk_jit_nodes(graph)):
            if node.kind() not in conversion_kinds:
                continue
            replacement = as_int_value(next(node.inputs()), node)
            if replacement is None:
                continue
            node.output().replaceAllUsesWith(replacement)
            node.destroy()
            changed += 1
        torch._C._jit_pass_dce(graph)
        folded += changed
        if changed == 0:
            break

    remaining = [
        n.kind()
        for n in _walk_jit_nodes(graph)
        if len(list(n.outputs())) == 1
        and str(next(n.outputs()).type()) in ("int", "float", "number", "Scalar", "bool")
        and n.kind() != "aten::size"
        and any("Tensor" in str(i.type()) for i in n.inputs())
    ]
    if remaining:
        raise RuntimeError(f"fold_shape_syncs left syncing conversions: {sorted(set(remaining))}")
    torch._C._jit_pass_lint(graph)
    return folded


class VPRInternal(nn.Module):
    __constants__ = ["run_h", "run_w", "use_rotated_descriptors"]

    def __init__(
        self,
        model: nn.Module,
        use_rotated_descriptors: bool,
        run_h: int,
        run_w: int,
    ):
        super().__init__()
        self.model = model
        self.use_rotated_descriptors = use_rotated_descriptors
        self.run_h = run_h
        self.run_w = run_w

    def forward(self, img: torch.Tensor):
        img = F.interpolate(
            img.half(),
            size=(self.run_h, self.run_w),
            mode="bilinear",
            align_corners=False,
        )

        if self.use_rotated_descriptors:
            img = torch.cat([
                img,
                torch.rot90(img, k=1, dims=[2, 3]),
                torch.rot90(img, k=2, dims=[2, 3]),
                torch.rot90(img, k=3, dims=[2, 3]),
            ], dim=0)

        features = self.model.backbone(img)
        features = self.model.aggregator.agg(features)
        linear = self.model.aggregator.linear
        features = linear(features.to(linear.weight.dtype))
        features = self.model.l2norm(features)
        return features.float()


class VPRModelWrapper:
    @torch.no_grad()
    def __init__(self, w, h, use_rotated_descriptors=True, use_cuda_graph=True):
        self.use_rotated_descriptors = use_rotated_descriptors
        self.run_w = 322
        self.run_h = 322
        self.batch = 1
        self.use_cuda_graph = use_cuda_graph and torch.cuda.is_available() and not torch.version.hip
        dummy = torch.zeros(self.batch, 3, h, w, device="cuda", dtype=torch.float)
        self.graph = None
        self.static_input = None
        self.static_output = None

        # Use MegaLoc instead of MixVPR for licensing reasons
        cache_path = os.path.join(
            "models",
            "cache",
            f"vpr_megaloc_halfsafe_{'rot' if use_rotated_descriptors else 'norot'}_b{self.batch}_{w}_{h}.pt",
        )

        self.model = None
        loaded_from_cache = False
        if os.path.exists(cache_path):
            try:
                self.model = torch.jit.load(cache_path, map_location="cuda").eval()
                loaded_from_cache = True
            except Exception:
                self.model = None

        if self.model is None:
            model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True)
            internal = VPRInternal(
                model.half().eval().cuda(),
                use_rotated_descriptors,
                self.run_h,
                self.run_w,
            ).eval()
            try:
                if self.use_cuda_graph:
                    self.model = torch.jit.trace(internal, (dummy,), check_trace=False).eval()
                else:
                    try:
                        self.model = torch.jit.script(internal).eval()
                        self.model(dummy)
                    except Exception:
                        self.model = torch.jit.trace(internal, (dummy,), check_trace=False).eval()
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.jit.save(self.model, cache_path)
            except Exception as exc:
                print(f"VPR JIT compilation disabled: {exc}")
                self.model = internal
                self.use_cuda_graph = False

        if self.use_cuda_graph and isinstance(self.model, torch.jit.ScriptModule):
            try:
                folded = self._fold_shape_syncs()
                if folded and loaded_from_cache:
                    torch.jit.save(self.model, cache_path)
            except Exception as exc:
                print(f"VPR shape-sync folding disabled: {exc}")
                self.use_cuda_graph = False

        for p in self.model.parameters():
            p.requires_grad_(False)

        if self.use_cuda_graph:
            try:
                self._capture_cuda_graph(dummy)
            except Exception as exc:
                print(f"VPR CUDA graph capture disabled: {exc}")
                self.graph = None
                self.static_input = None
                self.static_output = None
                if loaded_from_cache:
                    self.use_cuda_graph = False
                self.model(dummy)  # warmup
        else:
            self.model(dummy)  # warmup

    def _fold_shape_syncs(self) -> int:
        folded = 0
        for sub in self.model.modules():
            graph = getattr(getattr(sub, "forward", None), "graph", None)
            if graph is not None:
                folded += _fold_shape_syncs(graph)
        return folded

    @torch.no_grad()
    def _capture_cuda_graph(self, example: torch.Tensor):
        self.static_input = example.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.model(self.static_input)
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.model(self.static_input)

    @torch.no_grad()
    def __call__(self, img):
        if img.dim() == 3:
            img = img[None]
        assert img.shape[0] == self.batch, (
            f"VPRModelWrapper expects batch={self.batch}, got {img.shape[0]}"
        )
        if self.graph is not None:
            self.static_input.copy_(img)
            self.graph.replay()
            return self.static_output.clone()
        return self.model(img).clone()
