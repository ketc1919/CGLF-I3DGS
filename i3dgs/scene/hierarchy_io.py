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
import threading
import math

import torch
import numpy as np
import meshio

from utils import to_numpy


class _PlyVertexData:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, name):
        return self.data[name]


class _PlyData:
    def __init__(self, data, comments=None):
        self._vertex = _PlyVertexData(data)
        self.comments = comments or []

    def __getitem__(self, name):
        if name != "vertex":
            raise KeyError(f"Unsupported PLY element: {name}")
        return self._vertex


def _read_ply(path):
    mesh = meshio.read(path)
    fields = [
        ("x", mesh.points[:, 0].astype(np.float32, copy=False)),
        ("y", mesh.points[:, 1].astype(np.float32, copy=False)),
        ("z", mesh.points[:, 2].astype(np.float32, copy=False)),
    ]
    fields.extend((name, np.asarray(value)) for name, value in mesh.point_data.items())

    dtype = [(name, value.dtype, value.shape[1:]) for name, value in fields]
    data = np.empty(mesh.points.shape[0], dtype=dtype)
    for name, value in fields:
        data[name] = value
    return _PlyData(data)


def _write_ply(path, elements):
    points = np.stack([elements["x"], elements["y"], elements["z"]], axis=-1).astype(np.float32)
    point_data = {
        name: np.asarray(elements[name])
        for name in elements.dtype.names
        if name not in {"x", "y", "z"}
    }
    meshio.write(path, meshio.Mesh(points=points, cells=[], point_data=point_data), binary=True)


class HierarchyIOMixin:
    """Mixin providing PLY save/load functionality for HierarchyStructure."""

    PLY_SH_LAYOUT_COMMENT_PREFIX = "on-the-fly-sh-layout "
    PLY_SH_LAYOUT_COEFF_MAJOR = "coeff-major"
    PLY_SH_LAYOUT_CHANNEL_MAJOR = "channel-major"

    cpu_nodes_count: int
    cpu_nodes_capacity: int
    gaussian_nodes: dict
    active_gaussians: dict
    active_gaussians_count: int
    active_gaussians_capacity: int
    inference_mode: bool

    @classmethod
    def _flatten_f_rest_for_ply(cls, f_rest, layout):
        if f_rest.size == 0:
            return f_rest.reshape(f_rest.shape[0], 0)
        if layout == cls.PLY_SH_LAYOUT_COEFF_MAJOR:
            return f_rest.reshape(f_rest.shape[0], -1)
        if layout == cls.PLY_SH_LAYOUT_CHANNEL_MAJOR:
            return np.transpose(f_rest, (0, 2, 1)).reshape(f_rest.shape[0], -1)
        raise ValueError(f"Unsupported SH layout: {layout}")

    @classmethod
    def _restore_f_rest_from_ply(cls, f_rest_flat, num_rest, layout):
        n = f_rest_flat.shape[0]
        if f_rest_flat.size == 0:
            return np.empty((n, 0, 3), dtype=np.float32)
        if layout == cls.PLY_SH_LAYOUT_COEFF_MAJOR:
            return f_rest_flat.reshape(n, num_rest, 3)
        if layout == cls.PLY_SH_LAYOUT_CHANNEL_MAJOR:
            return f_rest_flat.reshape(n, 3, num_rest).transpose(0, 2, 1)
        raise ValueError(f"Unsupported SH layout: {layout}")

    @classmethod
    def _detect_ply_sh_layout(cls, plydata, field_names, child_fields):
        for comment in getattr(plydata, "comments", []):
            if comment.startswith(cls.PLY_SH_LAYOUT_COMMENT_PREFIX):
                layout = comment[len(cls.PLY_SH_LAYOUT_COMMENT_PREFIX):].strip()
                if layout in {cls.PLY_SH_LAYOUT_COEFF_MAJOR, cls.PLY_SH_LAYOUT_CHANNEL_MAJOR}:
                    return layout
        if child_fields:
            # Hierarchy PLYs are our internal save format; leaf/external PLYs follow the ecosystem default.
            return cls.PLY_SH_LAYOUT_COEFF_MAJOR
        return cls.PLY_SH_LAYOUT_CHANNEL_MAJOR

    def save_ply(self, path):
        """Save the hierarchy to a PLY file.

        Combines CPU nodes (gaussian_nodes) with active GPU nodes (active_gaussians)
        without modifying the hierarchy state:
          - Active nodes that already have an id_in_hierarchy override their CPU slot.
          - Active nodes with id_in_hierarchy == -1 are appended after the CPU nodes.
        """
        n_cpu = self.cpu_nodes_count
        n_active = self.active_gaussians_count

        if n_cpu == 0 and n_active == 0:
            print("No hierarchy nodes to save.")
            return

        # Identify which active nodes update an existing CPU slot vs. are new
        if n_active > 0:
            active_ids = self.active_gaussians["id_in_hierarchy"]["val"][:n_active].cpu()
            no_id_mask = (active_ids == -1).numpy()
            has_id_mask = ~no_id_mask
        else:
            active_ids = None
            no_id_mask = np.zeros(0, dtype=bool)
            has_id_mask = np.zeros(0, dtype=bool)

        n_new = int(no_id_mask.sum())
        n_total = n_cpu + n_new

        # Build attribute list and dtype
        sh_degree = int(np.sqrt(self.gaussian_nodes["f_rest"]["val"].shape[1] + 1)) - 1
        num_rest = (sh_degree + 1) ** 2 - 1
        num_children = self.gaussian_nodes["children"]["val"].shape[1]

        attributes = (
            ["x", "y", "z"]
            + [f"f_dc_{i}" for i in range(3)]
            + [f"f_rest_{i}" for i in range(num_rest * 3)]
            + ["opacity"]
            + [f"scale_{i}" for i in range(3)]
            + [f"rot_{i}" for i in range(4)]
            + ["kf_id"]
            + [f"child_{i}" for i in range(num_children)]
            + ["parent", "id_in_hierarchy"]
        )
        dtype_list = [
            (attr, "i4") if attr in {"kf_id", "parent", "id_in_hierarchy"} or attr.startswith("child_")
            else (attr, "f4")
            for attr in attributes
        ]

        # Start from a copy of the CPU nodes
        nodes = self.gaussian_nodes
        xyz        = to_numpy(nodes["xyz"]["val"][:n_cpu]).copy()
        f_dc       = to_numpy(nodes["f_dc"]["val"][:n_cpu]).reshape(n_cpu, 3).copy()
        f_rest     = to_numpy(nodes["f_rest"]["val"][:n_cpu]).copy()
        opacity    = to_numpy(nodes["opacity"]["val"][:n_cpu]).copy()
        scaling    = to_numpy(nodes["scaling"]["val"][:n_cpu]).copy()
        rotation   = to_numpy(nodes["rotation"]["val"][:n_cpu]).copy()
        kf_id      = to_numpy(nodes["kf_id"]["val"][:n_cpu]).copy()
        children   = to_numpy(nodes["children"]["val"][:n_cpu]).copy()
        parent     = to_numpy(nodes["parent"]["val"][:n_cpu]).copy()
        id_in_hier = to_numpy(nodes["id_in_hierarchy"]["val"][:n_cpu]).copy()

        # Override CPU slots with up-to-date values from active nodes
        if n_active > 0 and has_id_mask.any():
            ag = self.active_gaussians
            ids = active_ids[has_id_mask].numpy()
            xyz[ids]        = to_numpy(ag["xyz"]["val"][:n_active])[has_id_mask]
            f_dc[ids]       = to_numpy(ag["f_dc"]["val"][:n_active])[has_id_mask].reshape(-1, 3)
            f_rest[ids]     = to_numpy(ag["f_rest"]["val"][:n_active])[has_id_mask]
            opacity[ids]    = to_numpy(ag["opacity"]["val"][:n_active])[has_id_mask]
            scaling[ids]    = to_numpy(ag["scaling"]["val"][:n_active])[has_id_mask]
            rotation[ids]   = to_numpy(ag["rotation"]["val"][:n_active])[has_id_mask]
            kf_id[ids]      = to_numpy(ag["kf_id"]["val"][:n_active])[has_id_mask]
            children[ids]   = to_numpy(ag["children"]["val"][:n_active])[has_id_mask]
            parent[ids]     = to_numpy(ag["parent"]["val"][:n_active])[has_id_mask]
            id_in_hier[ids] = to_numpy(ag["id_in_hierarchy"]["val"][:n_active])[has_id_mask]

        # Append active nodes that have no CPU slot yet
        if n_new > 0:
            ag = self.active_gaussians
            xyz        = np.concatenate([xyz,        to_numpy(ag["xyz"]["val"][:n_active])[no_id_mask]])
            f_dc       = np.concatenate([f_dc,       to_numpy(ag["f_dc"]["val"][:n_active])[no_id_mask].reshape(-1, 3)])
            f_rest     = np.concatenate([f_rest,     to_numpy(ag["f_rest"]["val"][:n_active])[no_id_mask]])
            opacity    = np.concatenate([opacity,    to_numpy(ag["opacity"]["val"][:n_active])[no_id_mask]])
            scaling    = np.concatenate([scaling,    to_numpy(ag["scaling"]["val"][:n_active])[no_id_mask]])
            rotation   = np.concatenate([rotation,   to_numpy(ag["rotation"]["val"][:n_active])[no_id_mask]])
            kf_id      = np.concatenate([kf_id,      to_numpy(ag["kf_id"]["val"][:n_active])[no_id_mask]])
            children   = np.concatenate([children,   to_numpy(ag["children"]["val"][:n_active])[no_id_mask]])
            parent     = np.concatenate([parent,     to_numpy(ag["parent"]["val"][:n_active])[no_id_mask]])
            id_in_hier = np.concatenate([id_in_hier, np.arange(n_cpu, n_cpu + n_new, dtype=np.int32)])

        f_rest = self._flatten_f_rest_for_ply(f_rest, self.PLY_SH_LAYOUT_COEFF_MAJOR)

        # Pack into structured array and write PLY
        elements = np.empty(n_total, dtype=dtype_list)
        elements["x"] = xyz[:, 0];  elements["y"] = xyz[:, 1];  elements["z"] = xyz[:, 2]
        for i in range(3):
            elements[f"f_dc_{i}"] = f_dc[:, i]
        for i in range(num_rest * 3):
            elements[f"f_rest_{i}"] = f_rest[:, i]
        elements["opacity"] = opacity.squeeze(-1) if opacity.ndim > 1 else opacity
        for i in range(3):
            elements[f"scale_{i}"] = scaling[:, i]
        for i in range(4):
            elements[f"rot_{i}"] = rotation[:, i]
        elements["kf_id"] = kf_id
        for i in range(num_children):
            elements[f"child_{i}"] = children[:, i]
        elements["parent"] = parent
        elements["id_in_hierarchy"] = id_in_hier

        os.makedirs(os.path.dirname(path), exist_ok=True)
        _write_ply(path, elements)
        print(f"Saved hierarchy with {n_total} nodes ({n_cpu} CPU + {n_new} new active) to {path}")

    @staticmethod
    def _sort_numbered_fields(field_names, prefix):
        return sorted(
            (name for name in field_names if name.startswith(prefix)),
            key=lambda name: int(name[len(prefix):]),
        )

    @classmethod
    def _parse_ply_data(cls, plydata, sh_degree=None, num_neighbors_for_hierarchy=None):
        vertex = plydata["vertex"]
        field_names = vertex.data.dtype.names
        n = len(vertex["x"])

        dc_fields = cls._sort_numbered_fields(field_names, "f_dc_")
        rest_fields = cls._sort_numbered_fields(field_names, "f_rest_")
        scale_fields = cls._sort_numbered_fields(field_names, "scale_")
        rot_fields = cls._sort_numbered_fields(field_names, "rot_")
        child_fields = cls._sort_numbered_fields(field_names, "child_")

        if len(dc_fields) != 3:
            raise ValueError(f"Expected 3 DC SH fields, found {len(dc_fields)}")
        if len(rest_fields) % 3 != 0:
            raise ValueError(f"Expected f_rest_* fields to come in RGB triplets, found {len(rest_fields)}")
        if len(scale_fields) != 3:
            raise ValueError(f"Expected 3 scale fields, found {len(scale_fields)}")
        if len(rot_fields) != 4:
            raise ValueError(f"Expected 4 rotation fields, found {len(rot_fields)}")

        inferred_sh_degree = sh_degree
        num_rest = len(rest_fields) // 3
        if inferred_sh_degree is None:
            inferred_sh_degree = math.isqrt(num_rest + 1) - 1
        expected_num_rest = (inferred_sh_degree + 1) ** 2 - 1
        if num_rest != expected_num_rest:
            raise ValueError(
                f"Expected {(expected_num_rest * 3)} f_rest_* fields for SH degree {inferred_sh_degree}, "
                f"found {len(rest_fields)}"
            )

        inferred_neighbors = num_neighbors_for_hierarchy
        if inferred_neighbors is None:
            inferred_neighbors = len(child_fields) if child_fields else 4
        if child_fields and len(child_fields) != inferred_neighbors:
            raise ValueError(
                f"Expected {inferred_neighbors} child_* fields, found {len(child_fields)}"
            )

        sh_layout = cls._detect_ply_sh_layout(plydata, field_names, child_fields)

        # Extract xyz
        xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)

        # Extract f_dc
        f_dc = np.stack([vertex[name] for name in dc_fields], axis=-1)
        f_dc = f_dc.reshape(n, 1, 3)

        # Extract f_rest
        if rest_fields:
            f_rest = np.stack([vertex[name] for name in rest_fields], axis=-1).astype(np.float32)
            f_rest = cls._restore_f_rest_from_ply(f_rest, num_rest, sh_layout)
        else:
            f_rest = np.empty((n, 0, 3), dtype=np.float32)

        # Extract opacity
        opacity = vertex["opacity"].reshape(n, 1)

        # Extract scaling
        scaling = np.stack([vertex[name] for name in scale_fields], axis=-1)

        # Extract rotation
        rotation = np.stack([vertex[name] for name in rot_fields], axis=-1)

        # Extract kf_id
        if "kf_id" in field_names:
            kf_id = vertex["kf_id"].astype(np.int32)
        else:
            kf_id = np.full((n,), -1, dtype=np.int32)

        # Extract children
        if child_fields:
            children = np.stack([vertex[name] for name in child_fields], axis=-1).astype(np.int32)
        else:
            children = np.full((n, inferred_neighbors), -1, dtype=np.int32)

        # Extract parent
        if "parent" in field_names:
            parent = vertex["parent"].astype(np.int32)
        else:
            parent = np.full((n,), -1, dtype=np.int32)

        # Extract id_in_hierarchy
        if "id_in_hierarchy" in field_names:
            id_in_hierarchy = vertex["id_in_hierarchy"].astype(np.int32)
        else:
            id_in_hierarchy = np.arange(n, dtype=np.int32)

        return (
            {
                "xyz": xyz, "f_dc": f_dc, "f_rest": f_rest, "opacity": opacity,
                "scaling": scaling, "rotation": rotation, "kf_id": kf_id,
                "children": children, "parent": parent, "id_in_hierarchy": id_in_hierarchy,
            },
            inferred_sh_degree,
            inferred_neighbors,
        )

    @staticmethod
    def _load_root_nodes(hierarchy, data, n):
        root_mask = data["parent"] == -1
        root_indices = np.where(root_mask)[0]
        n_roots = len(root_indices)

        if n_roots > 0:
            # Expand active_gaussians storage if needed
            hierarchy.active_gaussians_capacity = hierarchy.expand_storage(
                hierarchy.active_gaussians, 0, 0, n_roots, "cuda"
            )
            hierarchy.active_gaussians_count = n_roots

            # Copy root nodes to active_gaussians
            hierarchy.active_gaussians["xyz"]["val"][:n_roots] = torch.from_numpy(data["xyz"][root_indices]).float().cuda()
            hierarchy.active_gaussians["f_dc"]["val"][:n_roots] = torch.from_numpy(data["f_dc"][root_indices]).float().cuda()
            hierarchy.active_gaussians["f_rest"]["val"][:n_roots] = torch.from_numpy(data["f_rest"][root_indices]).float().cuda()
            hierarchy.active_gaussians["opacity"]["val"][:n_roots] = torch.from_numpy(data["opacity"][root_indices]).float().cuda()
            hierarchy.active_gaussians["scaling"]["val"][:n_roots] = torch.from_numpy(data["scaling"][root_indices]).float().cuda()
            hierarchy.active_gaussians["rotation"]["val"][:n_roots] = torch.from_numpy(data["rotation"][root_indices]).float().cuda()
            hierarchy.active_gaussians["kf_id"]["val"][:n_roots] = torch.from_numpy(data["kf_id"][root_indices]).int().cuda()
            hierarchy.active_gaussians["children"]["val"][:n_roots] = torch.from_numpy(data["children"][root_indices]).int().cuda()
            hierarchy.active_gaussians["parent"]["val"][:n_roots] = torch.from_numpy(data["parent"][root_indices]).int().cuda()
            hierarchy.active_gaussians["id_in_hierarchy"]["val"][:n_roots] = torch.from_numpy(data["id_in_hierarchy"][root_indices]).int().cuda()

        return n_roots

    @classmethod
    def from_ply(
        cls,
        path,
        sh_degree=None,
        num_neighbors_for_hierarchy=None,
        hierarchy_max_screen_size=2.0,
        hierarchy_screen_size_threshold=0.5,
        hierarchy_cam_dist_threshold=1.0,
        hierarchy_recent_kf_skip=50,
        hierarchy_merge_ratio=0.1,
        hierarchy_merge_min_count=200_000,
    ):
        """Load a HierarchyStructure from a PLY file."""
        plydata = _read_ply(path)
        n = len(plydata["vertex"]["x"])

        data, sh_degree, num_neighbors_for_hierarchy = cls._parse_ply_data(
            plydata,
            sh_degree=sh_degree,
            num_neighbors_for_hierarchy=num_neighbors_for_hierarchy,
        )

        # Create args-like object for initialization
        class Args:
            pass
        args = Args()
        args.sh_degree = sh_degree
        args.num_neighbors_for_hierarchy = num_neighbors_for_hierarchy
        args.position_lr = 0.0
        args.feature_lr = 0.0
        args.scaling_lr = 0.0
        args.rotation_lr = 0.0
        args.opacity_lr = 0.0
        args.hierarchy_max_screen_size = hierarchy_max_screen_size
        args.hierarchy_screen_size_threshold = hierarchy_screen_size_threshold
        args.hierarchy_cam_dist_threshold = hierarchy_cam_dist_threshold
        args.hierarchy_recent_kf_skip = hierarchy_recent_kf_skip
        args.hierarchy_merge_ratio = hierarchy_merge_ratio
        args.hierarchy_merge_min_count = hierarchy_merge_min_count

        # Create instance in inference mode
        hierarchy = cls(args, threading.Lock(), inference_mode=True)

        # Populate gaussian_nodes
        hierarchy.cpu_nodes_capacity = hierarchy.expand_storage(
            hierarchy.gaussian_nodes, 0, 0, n, "cpu"
        )
        hierarchy.cpu_nodes_count = n

        for key in data:
            if key in ["kf_id", "children", "parent", "id_in_hierarchy"]:
                hierarchy.gaussian_nodes[key]["val"][:n] = torch.from_numpy(data[key]).int()
            else:
                hierarchy.gaussian_nodes[key]["val"][:n] = torch.from_numpy(data[key]).float()

        # Initialize active_gaussians with root nodes (those without parents)
        n_roots = cls._load_root_nodes(hierarchy, data, n)

        print(f"Loaded hierarchy with {n} nodes from {path}, {n_roots} root nodes as initial active set")

        return hierarchy
