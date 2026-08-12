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

import torch

from diff_gaussian_rasterization import adamUpdate, adamUpdateBasic
class BaseAdam:
    """Adam optimizer for regular parameters. This is simpler than torch.optim.Adam and initializes faster."""
    @torch.no_grad()
    def __init__(self, params, betas=(0.9, 0.999), eps=1e-15):
        self.params = params
        self.betas = betas
        self.eps = eps
        for param in self.params.values():
            if "exp_avg" not in param and "lr" in param:
                param["exp_avg"] = torch.zeros_like(
                    param["val"], memory_format=torch.preserve_format
                )
                param["exp_avg_sq"] = torch.zeros_like(
                    param["val"], memory_format=torch.preserve_format
                )

    def zero_grad(self):
        for param in self.params.values():
            param["val"].grad = None

    @torch.no_grad()
    def step(self):
        for param_dict in self.params.values():
            if "lr" not in param_dict:
                continue
            lr = param_dict["lr"]
            param = param_dict["val"]
            if param.grad is None:
                continue

            exp_avg = param_dict["exp_avg"]
            exp_avg_sq = param_dict["exp_avg_sq"]
            adamUpdateBasic(
                param,
                param.grad,
                exp_avg,
                exp_avg_sq,
                lr,
                self.betas[0],
                self.betas[1],
                self.eps,
            )
class SparseGaussianAdam(BaseAdam):
    """Adam optimizer for primitive parameters that can be optimized with sparse updates."""
    def __init__(self, params, betas=(0.9, 0.999), eps=1e-15, lr_dict={}):
        super().__init__(params=params, betas=betas, eps=eps)

        self.lr_dict = lr_dict
        # Convert learning rates to tensors
        for key, param in self.params.items():
            if "lr" not in param:
                continue
            if key not in self.lr_dict:
                param["lr"] = torch.tensor(
                    param["lr"], dtype=torch.float, device="cuda"
                )
            else:
                param["lr"] = torch.empty(0, dtype=torch.float, device="cuda")

    @torch.no_grad()
    def step(self, visibility, N):
        for key, param_dict in self.params.items():
            if "lr" not in param_dict or param_dict["lr"].numel() == 0:
                continue

            lr = param_dict["lr"]
            param = param_dict["val"]
            if param.grad is None:
                continue

            exp_avg = param_dict["exp_avg"]
            exp_avg_sq = param_dict["exp_avg_sq"]
            if param.dim() > 1:
                M = param.shape[1:].numel()
            else:
                M = 1
            adamUpdate(
                param,
                param.grad,
                exp_avg,
                exp_avg_sq,
                visibility,
                lr,
                self.betas[0],
                self.betas[1],
                self.eps,
                N,
                M,
            )

            # Update the learning rate
            if key in self.lr_dict:
                param_dict["lr"][:N][visibility] *= self.lr_dict[key]["lr_decay"]
                param_dict["lr"][:N].clamp_min_(self.lr_dict[key]["lr_init"] * 0.1)
