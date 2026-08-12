import threading
import torch
import gc
import torch.nn.functional as F

from utils import inverse_sigmoid, quat_to_rotmat, free_cuda_memory
from simple_knn._C import distIndex2

from scene.hierarchy_io import HierarchyIOMixin
from scene.hierarchy_cut import HierarchyCutMixin

def init_nodes(args, device):
    return {
        "xyz": { "val": torch.empty(0, 3, device=device)},
        "f_dc": { "val": torch.empty(0, 1, 3, device=device)},
        "f_rest": { "val":
            torch.empty(0,(args.sh_degree + 1) * (args.sh_degree + 1) - 1, 3, device=device)},
        "scaling": { "val": torch.empty(0, 3, device=device)},
        "rotation": { "val": torch.empty(0, 4, device=device)},
        "opacity": { "val": torch.empty(0, 1, device=device)},
        "kf_id": { "val": torch.empty(0, dtype=torch.int32, device=device)},
        "children": { "val": torch.empty(0, args.num_neighbors_for_hierarchy, dtype=torch.int32, device=device)},
        "parent": { "val": torch.empty(0, dtype=torch.int32, device=device)},
        "id_in_hierarchy": { "val": torch.empty(0, dtype=torch.int32, device=device)},
    }

class HierarchyStructure(HierarchyIOMixin, HierarchyCutMixin):
    """
    Represents a hierarchical structure for managing Gaussian parameters.
    Contains both active_gaussians (GPU) and inactive_gaussians (CPU) for hierarchy management.

    Functionality is organized into mixins:
        - HierarchyIOMixin (scene/hierarchy_io.py): PLY save/load
        - HierarchyCutMixin (scene/hierarchy_cut.py): Screen-space cut management (refine/coarsen)
    """
    PRIMITIVE_KEYS = ("xyz", "f_dc", "f_rest", "scaling", "rotation", "opacity")
    OPTIM_STATE_ATTRS = ("exp_avg", "exp_avg_sq")

    def __init__(self, args, lock, inference_mode):
        self.num_neighbors_for_hierarchy = args.num_neighbors_for_hierarchy
        self.hierarchy_max_screen_size = args.hierarchy_max_screen_size
        self.hierarchy_screen_size_threshold = args.hierarchy_screen_size_threshold
        self.hierarchy_cam_dist_threshold = args.hierarchy_cam_dist_threshold
        self.hierarchy_recent_kf_skip = args.hierarchy_recent_kf_skip
        self.hierarchy_merge_ratio = args.hierarchy_merge_ratio
        self.hierarchy_merge_min_count = args.hierarchy_merge_min_count
        self.lock = lock
        self.update_lock = threading.Lock()
        self.init_internal(args, inference_mode)


    def init_internal(self, args, inference_mode):
        self.update_thread = None
        self.new_gaussians, self.kept_mask = None, None
        self.n_updates = 0
        self.last_hierarchy_creation_kf_index = None
        self.inference_mode = inference_mode

        # Hierarchy parameters
        self.gaussian_nodes = init_nodes(args, "cpu")
        self.cpu_nodes_count = 0
        self.cpu_nodes_capacity = 0
        self.cpu_nodes_capacity = self.expand_storage(self.gaussian_nodes, 0, 0, 10_000_000, "cpu")

        ## Initialize active Gaussians
        self.active_gaussians = init_nodes(args, "cuda")
        self.active_gaussians["xyz"]["lr"] = args.position_lr
        self.active_gaussians["f_dc"]["lr"] = args.feature_lr
        self.active_gaussians["f_rest"]["lr"] = args.feature_lr / 20.0
        self.active_gaussians["scaling"]["lr"] = args.scaling_lr
        self.active_gaussians["rotation"]["lr"] = args.rotation_lr
        self.active_gaussians["opacity"]["lr"] = args.opacity_lr

        self.active_gaussians_count = 0
        self.active_gaussians_capacity = 0

        # Optimizer state for the gaussian primitives.
        if not self.inference_mode:
            for key in self.PRIMITIVE_KEYS:
                val = self.active_gaussians[key]["val"]
                val.requires_grad_(True)
                self.active_gaussians[key]["exp_avg"] = torch.zeros_like(val)
                self.active_gaussians[key]["exp_avg_sq"] = torch.zeros_like(val)

        # Pre-allocate active gaussians
        self.active_gaussians_capacity = self.expand_storage(self.active_gaussians, 0, 0, 2_000_000, "cuda")

    def release_nodes(self, nodes):
        for node in nodes.values():
            for attr in list(node.keys()):
                value = node.pop(attr)
                if isinstance(value, torch.Tensor):
                    value.grad = None
                    del value

    @torch.no_grad()
    def reset(self, args, inference_mode):
        if self.update_thread is not None and self.update_thread.is_alive():
            self.update_thread.join()

        with self.update_lock:
            with self.lock:
                self.release_nodes(self.gaussian_nodes)
                self.release_nodes(self.active_gaussians)
                self.init_internal(args, inference_mode)

        free_cuda_memory()

    # --- Storage Management ---

    def expand_storage(self, nodes, current_capacity, current_count, required_capacity, device):
        if current_capacity >= required_capacity:
            return current_capacity

        new_capacity = max(required_capacity, int(current_capacity * 1.5))

        for key in nodes:
            if "val" in nodes[key]:
                nodes[key]["val"].grad = None
                old_tensor = nodes[key]["val"]
                new_tensor = torch.empty((new_capacity, *old_tensor.shape[1:]), dtype=old_tensor.dtype, device=device)

                if current_count > 0:
                    with torch.no_grad():
                        new_tensor[:current_count] = old_tensor[:current_count]

                if isinstance(old_tensor, torch.nn.Parameter):
                    new_tensor = torch.nn.Parameter(new_tensor)
                    new_tensor.requires_grad = old_tensor.requires_grad
                elif old_tensor.requires_grad:
                    new_tensor.requires_grad = True

                # Aggressive cleanup of old tensor
                old_tensor.detach_()
                del old_tensor, nodes[key]["val"]
                nodes[key]["val"] = new_tensor

                for attr in [*self.OPTIM_STATE_ATTRS, "lr"]:
                    if attr in nodes[key] and isinstance(nodes[key][attr], torch.Tensor) and nodes[key][attr].ndim > 0:
                        nodes[key]["val"].grad = None
                        old_tensor = nodes[key][attr]
                        if not self.inference_mode:
                            new_tensor = torch.empty((new_capacity, *old_tensor.shape[1:]), dtype=old_tensor.dtype, device=device)
                            if current_count > 0:
                                with torch.no_grad():
                                    new_tensor[:current_count] = old_tensor[:current_count]
                        del old_tensor, nodes[key][attr]
                        nodes[key][attr] = new_tensor

        gc.collect()
        return new_capacity

    def add_and_prune(self, new_gaussians, kept_mask, buffer="active"):
        if buffer == "active":
            nodes = self.active_gaussians
            current_count = self.active_gaussians_count
            current_capacity = self.active_gaussians_capacity
            device = "cuda"
        elif buffer == "back":
            nodes = self.back_gaussians
            current_count = self.back_gaussians_count
            current_capacity = self.back_gaussians_capacity
            device = "cuda"
        else:
            raise ValueError("buffer must be 'active' or 'back'")

        n_new = list(new_gaussians.values())[0].shape[0]
        n_removed = (~kept_mask).sum().item()
        new_total = current_count - n_removed + n_new

        new_capacity = self.expand_storage(
            nodes, current_capacity, current_count, new_total, device
        )

        # Hole-filling: keep the buffer packed after removal.
        # The 'stable' region is [0, current_count - n_removed].
        # The 'tail' region is [current_count - n_removed, current_count].
        # Holes inside the stable region are filled by valid items from the tail;
        # holes inside the tail are simply truncated.
        src_indices = None
        dst_indices = None

        if n_removed > 0:
            cutoff = current_count - n_removed
            remove_mask = ~kept_mask
            dst_indices = remove_mask[:cutoff].nonzero().flatten()
            src_indices = (~remove_mask[cutoff:current_count]).nonzero().flatten() + cutoff

        for key in nodes:
            node = nodes[key]

            def apply_swap_and_append(tensor, attr_name, new_val=None, is_optim_state=False):
                if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
                    return
                if n_removed > 0 and src_indices.numel() > 0:
                    tensor[dst_indices] = tensor[src_indices]
                if n_new > 0:
                    start = current_count - n_removed
                    end = start + n_new
                    if new_val is not None:
                        tensor[start:end] = new_val
                    elif is_optim_state:
                        tensor[start:end] = 0

            if "val" in node:
                new_data = new_gaussians.get(key, None)
                apply_swap_and_append(node["val"], "val", new_val=new_data)

            if not self.inference_mode:
                for attr in self.OPTIM_STATE_ATTRS:
                    if attr in node:
                        apply_swap_and_append(node[attr], attr, is_optim_state=True)

        if buffer == "active":
            self.active_gaussians_count = new_total
            self.active_gaussians_capacity = new_capacity
        elif buffer == "back":
            self.back_gaussians_count = new_total
            self.back_gaussians_capacity = new_capacity

    # --- Properties ---

    @property
    def xyz(self):
        return self.active_gaussians["xyz"]["val"][:self.active_gaussians_count]

    @property
    def f_dc(self):
        return self.active_gaussians["f_dc"]["val"][:self.active_gaussians_count]

    @property
    def f_rest(self):
        return self.active_gaussians["f_rest"]["val"][:self.active_gaussians_count]

    @property
    def log_scaling(self):
        return self.active_gaussians["scaling"]["val"][:self.active_gaussians_count]

    def get_scaling(self,mask=None):
        if mask is not None:
            return torch.exp(self.active_gaussians["scaling"]["val"][:self.active_gaussians_count][mask, :])
        return torch.exp(self.active_gaussians["scaling"]["val"][:self.active_gaussians_count])

    @property
    def scaling(self):
        return self.get_scaling()

    def get_rotation(self,mask=None):
        if mask is not None:
            return F.normalize(self.active_gaussians["rotation"]["val"][:self.active_gaussians_count][mask, :])
        return F.normalize(self.active_gaussians["rotation"]["val"][:self.active_gaussians_count])

    @property
    def rotation(self):
        return self.get_rotation()

    @property
    def opacity(self):
        return torch.sigmoid(self.active_gaussians["opacity"]["val"][:self.active_gaussians_count])

    @property
    def kf_id(self):
        return self.active_gaussians["kf_id"]["val"][:self.active_gaussians_count]

    @property
    def children(self):
        return self.active_gaussians["children"]["val"][:self.active_gaussians_count]

    @property
    def parent(self):
        return self.active_gaussians["parent"]["val"][:self.active_gaussians_count]

    @property
    def id_in_hierarchy(self):
        return self.active_gaussians["id_in_hierarchy"]["val"][:self.active_gaussians_count]

    @property
    def n_active_gaussians(self):
        return self.active_gaussians_count

    # --- Gaussian Merging ---

    def merge_gaussians_enclosing(self,
            xyz, log_scaling, rotation, w,sigma_trunc=3.0, use_full_decomposition=True
        ):
            """
            xyz:        [B, K, 3]
            log_scaling:[B, K, 3]
            rotation:   [B, K, 4]  (xyzw)
            returns:
                mu:         [B, 3]
                log_scale:  [B, 3]
            quat:      [B, 4]    (identity)
            """
            B, K, _ = xyz.shape
            device, dtype = xyz.device, xyz.dtype

            # --- merged mean ---
            mu = xyz.mean(dim=1)                                  # [B, 3]

            # --- scales ---
            scales = torch.exp(log_scaling)                       # [B, K, 3]
            var = scales ** 2                                     # [B, K, 3]

            # --- rotation matrices ---
            R = quat_to_rotmat(rotation)                          # [B, K, 3, 3]

            # --- exact diagonal of rotated covariance ---
            # diag(R Σ R^T)_d = sum_j R_dj^2 * s_j^2
            cov_diag = (R ** 2) @ var[..., None]                  # [B, K, 3, 1]
            cov_diag = cov_diag.squeeze(-1)                       # [B, K, 3]

            # --- mean offset term ---
            dmu = xyz - mu[:, None, :]                            # [B, K, 3]

            # --- 3σ truncation bound ---
            bound = (sigma_trunc ** 2) * cov_diag + dmu ** 2      # [B, K, 3]

            # --- enclosing variance ---
            merged_var = bound.max(dim=1).values                  # [B, 3]

            scale = torch.sqrt(torch.clamp(merged_var, min=1e-12))
            log_scale = torch.log(scale)

            quat = torch.zeros((B, 4), device=device, dtype=dtype)
            quat[:, 3] = 1.0                                      # identity

            return mu, log_scale, quat

    # --- Hierarchy Level Creation ---

    # create new level of hierarchy from given gaussian indices
    @torch.no_grad()
    def create_hierarchy_level(self, cam_centre, candidate_mask, dist_cam=False):
        if dist_cam:
            dist_cam = torch.linalg.vector_norm(
                self.xyz - cam_centre[None], dim=-1
            )
            screen_size = self.f * self.scaling.mean(dim=-1) / dist_cam
        else:
            screen_size = torch.ones_like(self.opacity).squeeze()

        xyz = self.xyz[candidate_mask].contiguous()

        selected_groups, idx_remapping = self._select_merge_groups(xyz, candidate_mask)
        selected_nn_idx = idx_remapping[selected_groups]

        merged_gaussians = self._compute_merged_gaussians(selected_nn_idx, cam_centre, screen_size)

        num_merged = self._link_hierarchy_and_store(selected_nn_idx, merged_gaussians)

        return num_merged

    @torch.no_grad()
    def _select_merge_groups(self, xyz, candidate_mask):
        k = self.num_neighbors_for_hierarchy - 1

        dist, nn_idx = distIndex2(xyz, k)
        nn_idx = nn_idx.view(-1, k)

        ## Removing duplicates, not ideal but needed for now
        N = nn_idx.shape[0]
        device = nn_idx.device

        remaining_mask = torch.ones(N, dtype=torch.bool, device=device)
        all_selected = []
        claims = torch.full((N,), -1, dtype=torch.long, device=device)

        for _ in range(10):
            candidates = remaining_mask.nonzero(as_tuple=True)[0]
            if candidates.shape[0] < k + 1:
                break

            perm = candidates[torch.randperm(candidates.shape[0], device=device)]
            idx = perm[: candidates.shape[0] // (k + 1)]
            groups = torch.cat([idx[:, None], nn_idx[idx]], dim=-1)

            # Check all indices in groups are still available
            all_available = remaining_mask[groups].all(dim=1)
            groups = groups[all_available]

            if groups.shape[0] == 0:
                continue

            # Conflict resolution
            group_ids = torch.arange(groups.shape[0], device=device)
            flat_groups = groups.view(-1)

            claims[flat_groups] = group_ids.repeat_interleave(groups.shape[1])
            valid_groups_mask = (claims[groups] == group_ids.unsqueeze(-1)).all(dim=1)
            good_groups = groups[valid_groups_mask]

            claims[flat_groups] = -1

            all_selected.append(good_groups)
            remaining_mask[good_groups.view(-1)] = False

        selected_groups = torch.cat(all_selected, dim=0)
        idx_remapping = torch.arange(self.active_gaussians_count, device=xyz.device)[candidate_mask]

        return selected_groups, idx_remapping

    @torch.no_grad()
    def _compute_merged_gaussians(self, selected_nn_idx, cam_centre, screen_size):
        k = self.num_neighbors_for_hierarchy - 1

        # Compute merging weights based on contribution to the rendering
        weights = self.active_gaussians["opacity"]["val"][:self.active_gaussians_count][
            selected_nn_idx, 0
        ].sigmoid() * (screen_size[selected_nn_idx] ** 2)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights.unsqueeze_(-1)

        # SH / opacity
        merged_f_dc = (
            self.f_dc[selected_nn_idx] * weights.unsqueeze(-1)
        ).sum(dim=1)

        merged_f_rest = (
            self.f_rest[selected_nn_idx] * weights.unsqueeze(-1)
        ).sum(dim=1)

        merged_opacity = inverse_sigmoid(
            (self.opacity[selected_nn_idx] * weights)
            .sum(dim=1)
        )

        merged_xyz_shared_mu = (self.xyz[selected_nn_idx] * weights).sum(dim=1)
        merged_log_scale = (self.scaling[selected_nn_idx] * weights * (k+1)).sum(dim=1).log()
        merged_quat = (self.rotation[selected_nn_idx] * weights).sum(dim=1)

        ## Clamp screen size (Merging small Gaussians should give us relatively small ones)
        dist = torch.linalg.vector_norm(
            merged_xyz_shared_mu - cam_centre[None], dim=-1
        )
        screen_size = self.f * torch.exp(merged_log_scale) / dist[..., None]
        max_screen_size = self.hierarchy_max_screen_size
        mask = screen_size > max_screen_size
        if mask.any():
            merged_log_scale[mask] = torch.log(
                max_screen_size * dist[..., None].expand(-1, 3)[mask] / self.f)

        return {
            "xyz": merged_xyz_shared_mu,
            "f_dc": merged_f_dc,
            "f_rest": merged_f_rest,
            "opacity": merged_opacity,
            "scaling": merged_log_scale,
            "rotation": merged_quat,
        }

    @torch.no_grad()
    def _link_hierarchy_and_store(self, selected_nn_idx, merged_gaussians):
        ## Building the hierarchy structure
        num_merged = merged_gaussians["xyz"].shape[0]
        device = self.active_gaussians["xyz"]["val"].device

        # Determine IDs for the new Parent nodes (stored after all current nodes)
        # Parents are always new, so they start at current cpu_nodes_count
        parent_ids = torch.arange(
            self.cpu_nodes_count,
            self.cpu_nodes_count + num_merged,
            dtype=torch.int32,
            device=device
        )

        # Link Children to Parents: Update the 'parent' field in active gaussians FIRST
        # selected_nn_idx is [num_merged, neighbors]. We broadcast parent_ids to match.
        self.parent[selected_nn_idx] = parent_ids.unsqueeze(-1)

        # Get current IDs for children being removed
        children_ids = self.id_in_hierarchy[selected_nn_idx].view(-1)

        # For children without IDs, assign them NOW in active Gaussians BEFORE storing
        children_no_id_mask = children_ids == -1
        num_new_children = 0
        if children_no_id_mask.any():
            num_new_children = children_no_id_mask.sum().item()
            new_children_ids = torch.arange(
                self.cpu_nodes_count + num_merged,
                self.cpu_nodes_count + num_merged + num_new_children,
                dtype=torch.int32,
                device=device
            )
            # Update the id_in_hierarchy in active Gaussians
            self.id_in_hierarchy[selected_nn_idx.view(-1)[children_no_id_mask]] = new_children_ids

        # Expand CPU storage for hierarchy
        self.cpu_nodes_capacity = self.expand_storage(
            self.gaussian_nodes,
            self.cpu_nodes_capacity,
            self.cpu_nodes_count,
            self.cpu_nodes_count + num_merged + num_new_children,
            "cpu"
        )

        # Create Parent Gaussians
        merged_gaussians["kf_id"] = self.kf_id[selected_nn_idx[:,0]]
        # Link Parents to Children: use the final IDs (now all valid)
        merged_gaussians["children"] = self.id_in_hierarchy[selected_nn_idx]
        merged_gaussians["parent"] = torch.full((num_merged,), -1, device=device, dtype=torch.int32)
        merged_gaussians["id_in_hierarchy"] = parent_ids

        # Store parent gaussians in hierarchy
        ids = parent_ids.cpu()
        for key in self.gaussian_nodes.keys():
            val = merged_gaussians[key]
            self.gaussian_nodes[key]["val"][ids] = val.cpu()

        # Store children Gaussians in hierarchy
        ids = self.id_in_hierarchy[selected_nn_idx.view(-1)].cpu()
        for key in self.gaussian_nodes.keys():
            val = self.active_gaussians[key]["val"][:self.active_gaussians_count][selected_nn_idx.view(-1)]
            self.gaussian_nodes[key]["val"][ids] = val.cpu()

        self.cpu_nodes_count += num_merged + num_new_children

        mask = torch.ones(self.active_gaussians_count, dtype=torch.bool, device=device)
        mask[selected_nn_idx.view(-1)] = False
        with self.lock:
            self.add_and_prune(merged_gaussians, mask)

        return num_merged

    # --- Inference Mode ---

    def enable_inference_mode(self):
        ## Put active Gaussians in the hierarchy
        if self.active_gaussians_count == 0:
            self.inference_mode = True
            return

        # Get the hierarchy IDs of active Gaussians
        active_ids = self.id_in_hierarchy

        num_to_update = (active_ids >= 0).sum().item()
        no_id_mask = active_ids == -1
        num_to_add = (no_id_mask).sum().item()

        # For Gaussians without IDs, assign them now in active Gaussians
        if num_to_add > 0:
            new_ids = torch.arange(
                self.cpu_nodes_count,
                self.cpu_nodes_count + num_to_add,
                dtype=torch.int32,
                device=active_ids.device
            )
            # Update the id_in_hierarchy in active Gaussians
            self.id_in_hierarchy[no_id_mask] = new_ids

            # Expand CPU storage
            self.cpu_nodes_capacity = self.expand_storage(
                self.gaussian_nodes,
                self.cpu_nodes_capacity,
                self.cpu_nodes_count,
                self.cpu_nodes_count + num_to_add,
                "cpu"
            )

        # Store all active Gaussians directly using their IDs
        ids = self.id_in_hierarchy.cpu()
        for key in self.gaussian_nodes.keys():
            val = self.active_gaussians[key]["val"][:self.active_gaussians_count]
            self.gaussian_nodes[key]["val"][ids] = val.cpu()

        self.cpu_nodes_count += num_to_add

        print(f"Inference mode enabled: updated {num_to_update} existing nodes, added {num_to_add} new nodes. Total hierarchy nodes: {self.cpu_nodes_count}")

        self.inference_mode = True
