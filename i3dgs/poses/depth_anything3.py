# Parts of this file (marked "From DA3 input_processor.py") are adapted from
# Depth Anything 3, https://github.com/ByteDance-Seed/Depth-Anything-3,
# used under the Apache License 2.0 (a copy is provided in licenses/Apache-2.0.txt).
# Modified to operate on GPU tensors.

import contextlib
import logging
import os
import sys
import warnings

import torch
import torch.nn.functional as F

# From DA3 input_processor.py
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
_DA3_PATCH_SIZE = 14


def load_depth_anything3_api():
    try:
        from depth_anything_3.api import DepthAnything3
        return DepthAnything3
    except ModuleNotFoundError as exc:
        repo_root = os.path.join(
            os.path.dirname(__file__),
            "..",
            "submodules",
            "Depth-Anything-3",
        )
        src_path = os.path.join(repo_root, "src")
        if os.path.isdir(src_path) and src_path not in sys.path:
            sys.path.append(src_path)
            try:
                from depth_anything_3.api import DepthAnything3
                return DepthAnything3
            except ModuleNotFoundError:
                pass
        raise RuntimeError(
            "Depth-Anything-3 package not found."
        ) from exc


def nearest_multiple(x: int, patch: int) -> int:
    down = (x // patch) * patch
    up = down + patch
    return up if abs(up - x) <= abs(x - down) else down


# From DA3 input_processor.py
def resize_for_da3(images: torch.Tensor, process_res: int = 504) -> torch.Tensor:
    images = images.float()
    _, _, h, w = images.shape

    scale = process_res / float(max(h, w))

    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, nearest_multiple(new_h, _DA3_PATCH_SIZE))
    new_w = max(1, nearest_multiple(new_w, _DA3_PATCH_SIZE))

    if (new_h, new_w) != (h, w):
        mode = "bicubic" if scale > 1.0 else "area"
        images = F.interpolate(images, size=(new_h, new_w), mode=mode, align_corners=False if mode == "bicubic" else None)

    return images


# From DA3 input_processor.py
def normalize_da3(images: torch.Tensor) -> torch.Tensor:
    return (images - _IMAGENET_MEAN) / _IMAGENET_STD


@contextlib.contextmanager
def _suppress_model_output():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield


class DepthAnything3:
    @torch.no_grad()
    def __init__(self, width, height, n_cams, da3_model_dir):
        self.n_cams = n_cams
        self.width = width
        self.height = height
        self.process_res = 504
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with _suppress_model_output():
            logging.getLogger("depth_anything_3").setLevel(logging.ERROR)
            DepthAnything3Api = load_depth_anything3_api()
            try:
                self.model = DepthAnything3Api.from_pretrained(da3_model_dir, local_files_only=True)
            except Exception:
                self.model = DepthAnything3Api.from_pretrained(da3_model_dir)
            self.model = self.model.to(device=self.device).eval()

        dummy = torch.rand([n_cams, 3, height, width], device=self.device)
        with _suppress_model_output():
            self(dummy)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor):
        images = images.to(self.device, torch.float32)
        if not images.is_contiguous():
            images = images.contiguous()

        _, _, h, w = images.shape
        scale = self.process_res / float(max(h, w))
        new_h = max(1, nearest_multiple(int(round(h * scale)), _DA3_PATCH_SIZE))
        new_w = max(1, nearest_multiple(int(round(w * scale)), _DA3_PATCH_SIZE))
        scale_h = h / float(new_h)
        scale_w = w / float(new_w)

        imgs = images
        if (new_h, new_w) != (h, w):
            mode = "bicubic" if scale > 1.0 else "area"
            imgs = F.interpolate(imgs, size=(new_h, new_w), mode=mode, align_corners=False if mode == "bicubic" else None)
        imgs = normalize_da3(imgs)

        with _suppress_model_output():
            raw_output = self.model(imgs[None], export_feat_layers=[])

        intrinsics = raw_output["intrinsics"]
        k_mean = intrinsics[0].mean(dim=0)
        raw_output["f"] = (k_mean[0, 0] * scale_w + k_mean[1, 1] * scale_h) / 2.0
        return raw_output
