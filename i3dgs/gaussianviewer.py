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

import json
import os
import numpy as np
from argparse import ArgumentParser, Namespace
from imgui_bundle import imgui_ctx, imgui, hello_imgui
from enum import IntEnum, auto
import time

from graphdecoviewer import Viewer
from graphdecoviewer.types import ViewerMode
from graphdecoviewer.widgets.image import TorchImage
from graphdecoviewer.widgets.radio import RadioPicker
from graphdecoviewer.widgets.cameras.fps import FPSCamera
from graphdecoviewer.widgets.ellipsoid_viewer import EllipsoidViewer

class Dummy(object):
    pass


class CompressedTorchImage(TorchImage):
    """
    TorchImage that JPEG-compresses the frame before sending it to the client
    in server/client mode. Compression settings are driven by the viewer
    through `compression_enabled` and `jpeg_quality`.
    """
    def __init__(self, mode: ViewerMode):
        super().__init__(mode)
        self.compression_enabled = False
        self.jpeg_quality = 85

    def server_send(self):
        binary, text = super().server_send()
        if binary is None or not self.compression_enabled:
            return binary, text
        image = np.frombuffer(binary, dtype=np.uint8).reshape(text["shape"])
        success, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                        [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
        if not success:
            return binary, text
        return memoryview(encoded), {**text, "jpeg": True}

    def client_recv(self, binary, text):
        if text is not None and text.get("jpeg", False):
            import cv2
            image = cv2.imdecode(np.frombuffer(binary, dtype=np.uint8), cv2.IMREAD_COLOR)
            image = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            binary = memoryview(image.reshape(-1))
            text = {"shape": tuple(image.shape)}
        super().client_recv(binary, text)


class SnapMode(IntEnum):
    free = auto()
    keyframe = auto()
    last = auto()

class GaussianViewer(Viewer):
    atttrs_to_sync = [
            "render_mode_id", "draw_poses", "draw_gt_poses", "pose_sizes",
            "scaling_factor", "bg_color",
            "show_top_view", "keyframe_id", "altitude_control", "altitude_smoothing",
            "snap_to_closest", "snap_mode_id", "next_keyframe", "prev_keyframe",
            "reset_intrinsics_flag",
            "render_tau", "jpeg_compression", "jpeg_quality",
            "keep_alive",
        ]

    def __init__(self, mode: ViewerMode):
        super().__init__(mode)
        self.window_title = "Gaussian Viewer"
        self.throttling = False
        self.keep_alive = False
        self.default_layout_pending = None

    def import_server_modules(self):
        global torch
        import torch

        global cv2
        import cv2

        global SceneModel
        from scene.scene_model import SceneModel

        global draw_poses
        from utils import draw_poses

    @classmethod
    def from_scene(cls, scene_source: str, mode: ViewerMode, args: Namespace):
        viewer = cls(mode)
        viewer.scene_model = SceneModel.from_scene(scene_source, args)
        return viewer
    
    @classmethod
    def from_scene_model(cls, scene_model: 'SceneModel', mode: ViewerMode):
        viewer = cls(mode)
        viewer.scene_model = scene_model
        return viewer

    def create_widgets(self):
        if self.mode is not ViewerMode.CLIENT:
            width = self.scene_model.width
            height = self.scene_model.height
            fov_y = np.rad2deg(self.scene_model.FoVy)
            self.num_keyframes = len(self.scene_model.keyframes)
            self.render_tau = self.scene_model.render_tau
        else:
            width, height, fov_y = 2, 2, 1
            self.num_keyframes = 0
            self.render_tau = 1.0
        self.point_view_camera = FPSCamera(self.mode, width, height, fov_y, 0.01, 100)
        self.top_view_camera = FPSCamera(self.mode, 480, 480, 60, 0.01, 100,
            to_world=np.array([[-1, 0, 0, 0],
                               [0, 0.7071, 0.7071, -3],
                               [0, -0.7071, 0.7071, -2],
                               [0, 0, 0, 1]])
        )
        self.cameras = {"top_view": self.top_view_camera, "point_view": self.point_view_camera}
        self.point_view = CompressedTorchImage(self.mode)
        self.top_view = CompressedTorchImage(self.mode)
        self.views = {"top_view": self.top_view, "point_view": self.point_view}
        self.ellipsoid_viewer = EllipsoidViewer(self.mode)

        # Render modes
        self.render_modes = ["Splats", "Depth", "Ellipsoids"]
        self.render_mode_id = 0

        # Render settings
        views = ["top_view", "point_view"]
        self.draw_poses = {view: False for view in views}
        self.draw_gt_poses = {view: False for view in views}
        self.pose_sizes = {view: 0.1 for view in views}
        self.scaling_factor = {"top_view": 0.002, "point_view": 1}
        self.reset_intrinsics_flag = {view: False for view in views}
        self.bg_color = [0.0, 0.0, 0.0]
        self.show_top_view = False
        self.max_fps = 20
        self.jpeg_compression = False
        self.jpeg_quality = 85
        self.last_show_gui_time = time.time()
        self.only_align_gt_to_first_x = False

        # Camera settings
        self.keyframe_id = 0
        self.reset_pose = False
        self.altitude_control = False
        self.altitude_smoothing = 0.9
        self.snap_to_closest = False
        self.snap_mode = RadioPicker(ViewerMode.LOCAL, SnapMode.free)
        self.snap_mode_id = int(self.snap_mode.value)
        self.next_keyframe = False
        self.prev_keyframe = False
        self.updated_pose = None


    def render_mode(self):
        return self.render_modes[self.render_mode_id]

    def set_snap_mode(self, value):
        value = SnapMode(value)
        if self.snap_mode.value != value:
            self.snap_mode.states[self.snap_mode.value] = False
            self.snap_mode.states[value] = True
            self.snap_mode.value = value
        self.snap_mode_id = int(value)

    def reset_intrinsics(self, view):
        camera = self.cameras[view]
        camera.res_x = self.scene_model.width // 2 if view == "top_view" else self.scene_model.width
        camera.fov_x = self.scene_model.FoVx
        camera.res_y = self.scene_model.height // 2 if view == "top_view" else self.scene_model.height
        camera.fov_y = self.scene_model.FoVy

    def onconnect(self, websocket):
        if self.mode == ViewerMode.SERVER:
            websocket.send(json.dumps({
                "num_keyframes": self.num_keyframes,
                "width": self.point_view_camera.res_x,
                "height": self.point_view_camera.res_y,
                "fov_y": self.point_view_camera.fov_y,
                "ellipsoid_enabled": self.ellipsoid_viewer.enabled,
            }), text=True)
        if self.mode == ViewerMode.CLIENT:
            data = json.loads(websocket.recv())
            self.num_keyframes = data["num_keyframes"]
            self.point_view_camera.res_x = data["width"]
            self.point_view_camera.res_y = data["height"]
            self.point_view_camera.fov_y = data["fov_y"]
            self.point_view_camera.compute_fov_x()
            self.ellipsoid_viewer.enabled = data["ellipsoid_enabled"]
            self.keep_alive = data.get("keep_alive", self.keep_alive)
    
    def step(self):
        self.set_snap_mode(self.snap_mode_id)
        # Get camera matrix
        self.updated_pose = None
        if self.snap_mode.value in [SnapMode.keyframe, SnapMode.last]:
            self.num_keyframes = len(self.scene_model.keyframes)
            if self.num_keyframes == 0:
                self.snap_mode.value = SnapMode.free
                self.next_keyframe = False
                self.prev_keyframe = False
            else:
                if self.next_keyframe:
                    self.keyframe_id = min(self.num_keyframes - 1, self.keyframe_id + 1)
                if self.prev_keyframe:
                    self.keyframe_id = max(0, self.keyframe_id - 1)
                keyframe_id = self.keyframe_id if self.snap_mode.value == SnapMode.keyframe else -1
                point_viewmatrix = self.scene_model.keyframes[keyframe_id].get_Rt()
                self.updated_pose = torch.linalg.inv(point_viewmatrix.detach()).cpu().numpy()
        else:
            if self.altitude_control and len(self.scene_model.keyframes) > 0:
                camera_position = torch.tensor(self.point_view_camera.origin, dtype=torch.float32).cuda()
                n_closest = 4
                closest_keyframes = self.scene_model.get_closest_keyframe(camera_position, n_closest)
                mean_closest_altitude = (torch.stack([kf.approx_centre for kf in closest_keyframes]).sum(axis=0) / n_closest)[1]

                if abs(mean_closest_altitude - self.point_view_camera.origin[1]) > 1e-4:
                    dist = mean_closest_altitude - self.point_view_camera.origin[1]
                    to_world = self.point_view_camera.to_world.copy()
                    to_world[1, 3] += (1.0 - self.altitude_smoothing) * self.point_view_camera.speed * dist
                    self.updated_pose = to_world
            if self.snap_to_closest and len(self.scene_model.keyframes) > 0:
                camera_position = torch.tensor(self.point_view_camera.origin, dtype=torch.float32).cuda()
                closest_keyframe = self.scene_model.get_closest_keyframe(camera_position)[0]
                keyframe_pose = torch.linalg.inv(closest_keyframe.get_Rt()).detach().cpu().numpy()
                self.updated_pose = keyframe_pose

        # Sync render_tau to scene_model
        self.scene_model.render_tau = self.render_tau

        # Sync compression settings to the image widgets
        for image_view in self.views.values():
            image_view.compression_enabled = self.jpeg_compression
            image_view.jpeg_quality = self.jpeg_quality

        # Render scene
        for view in ["point_view", "top_view"]:
            camera = self.cameras[view]
            if self.reset_intrinsics_flag[view]:
                self.reset_intrinsics(view)

            # Draw Gaussians
            if (view == "point_view" and self.render_mode() in ["Splats", "Depth"]) or (view == "top_view" and self.show_top_view):
                width = camera.res_x
                height = camera.res_y
                viewmatrix = torch.tensor(camera.to_camera, dtype=torch.float32).cuda().transpose(0, 1)
                render_pkg = self.scene_model.render(width, height, viewmatrix, self.scaling_factor[view], torch.tensor(self.bg_color, device="cuda"), view=="top_view", 
                                                     camera.fov_x, camera.fov_y)
                if self.render_mode() == "Splats":
                    image = render_pkg["render"].clamp(0, 1.0).mul(255).permute(1, 2, 0).byte().contiguous()
                elif self.render_mode() == "Depth":
                    image = render_pkg["invdepth"][0].mul(100).clamp(0, 255).byte().cpu().numpy()
                    image = cv2.cvtColor(cv2.applyColorMap(image, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
                    image = torch.tensor(image).cuda()

                # Draw overlays
                if self.draw_poses[view] or self.draw_gt_poses[view] or view == "top_view":
                    image = image.contiguous().cpu().numpy()
                    common_opt = (image, viewmatrix, camera.fov_x, self.pose_sizes[view], self.scene_model.width, self.scene_model.height)
                    if self.draw_gt_poses[view]:
                        image = draw_poses(*common_opt, self.scene_model.get_gt_keyframe_Rts(True, self.scene_model.args.num_keyframes_miniba_bootstrap if self.only_align_gt_to_first_x else 0), self.scene_model.gt_f, (255, 0, 0))
                    if self.draw_poses[view]:
                        Rts = self.scene_model.get_keyframe_Rts()
                        image = draw_poses(*common_opt, Rts, self.scene_model.f, (255, 255, 255))
                        if self.scene_model.prev_keyframes_odo is not None:
                            prev_ids = [keyframe.index for keyframe in self.scene_model.prev_keyframes_odo]
                            image = draw_poses(*common_opt, self.scene_model.get_keyframe_Rts(prev_ids), self.scene_model.f, (0, 255, 255))
                        if self.scene_model.prev_keyframes_loop is not None:
                            prev_ids = [keyframe.index for keyframe in self.scene_model.prev_keyframes_loop]
                            image = draw_poses(*common_opt, self.scene_model.get_keyframe_Rts(prev_ids), self.scene_model.f, (100, 255, 100))
                        if len(self.scene_model.keyframes) > 0:
                            image = draw_poses(*common_opt, self.scene_model.get_keyframe_Rts([len(self.scene_model.keyframes) - 1]), self.scene_model.f, (255, 30, 255))
                    if view == "top_view":
                        point_viewmatrix = torch.tensor(self.point_view_camera.to_camera, dtype=torch.float32).cuda()[None]
                        image = draw_poses(*common_opt, point_viewmatrix, self.scene_model.f, (0, 255, 255))
                    image = torch.tensor(image).cuda()

                # Update the buffer
                self.views[view].step(image)

            # Draw ellipsoids
            elif self.render_mode() == "Ellipsoids" and view == "point_view":
                self.ellipsoid_viewer.step(camera)

    def setup_default_layout(self):
        """
        Build the default docking layout: settings panels stacked in a left
        column, Point View filling the remaining space. Sizes are expressed
        relative to the viewport and font size so the layout adapts to any
        resolution and DPI scaling.
        """
        dockspace_id = hello_imgui.get_runner_params().docking_params.dock_space_id_from_name("MainDockSpace")
        if dockspace_id is None:
            return False
        # Left column sized to fit the settings widgets (~16 text lines wide).
        settings_ratio = min(0.3, hello_imgui.em_size(16) / imgui.get_main_viewport().size.x)
        imgui.internal.dock_builder_remove_node_child_nodes(dockspace_id)
        _, settings_id, main_id = imgui.internal.dock_builder_split_node_py(
            dockspace_id, imgui.Dir.left, settings_ratio)
        _, top_view_settings_id, point_view_settings_id = imgui.internal.dock_builder_split_node_py(
            settings_id, imgui.Dir.down, 0.35)
        imgui.internal.dock_builder_dock_window("Point View", main_id)
        imgui.internal.dock_builder_dock_window("Point View Settings", point_view_settings_id)
        imgui.internal.dock_builder_dock_window("Top View Settings", top_view_settings_id)
        imgui.internal.dock_builder_finish(dockspace_id)
        return True

    def show_gui(self):
        if self.default_layout_pending is None:
            # Only build the default layout when HelloImGui has no saved
            # settings, so layouts customized by the user are kept.
            ini_location = hello_imgui.ini_settings_location(hello_imgui.get_runner_params())
            self.default_layout_pending = ini_location is None or not os.path.isfile(ini_location)
        if self.default_layout_pending and imgui.get_frame_count() > 1:
            # Wait for frame 2: HelloImGui wipes the dockspace child nodes at
            # the start of the second frame (before this callback), which
            # would discard a layout built on the first frame.
            self.default_layout_pending = not self.setup_default_layout()

        if self.updated_pose is not None:
            # Snaps are hard camera transitions: stale FPS inertia must not move
            # the camera away from the requested pose on the following frames.
            self.point_view_camera.origin_motion.fill(0)
            self.point_view_camera.rotation_motion.fill(0)
            self.point_view_camera.smoothed_origin_motion.fill(0)
            self.point_view_camera.smoothed_rotation_motion.fill(0)
            self.point_view_camera.update_pose(self.updated_pose)
            self.updated_pose = None

        with imgui_ctx.begin(f"Point View Settings"):
            render_modes = self.render_modes.copy()
            if not self.ellipsoid_viewer.enabled:
                render_modes.remove("Ellipsoids")

            _, self.render_mode_id = imgui.list_box("Render Mode", self.render_mode_id, render_modes)

            imgui.separator_text("Render Settings")
            if self.render_mode() in ["Splats", "Depth"]:
                _, self.scaling_factor["point_view"] = imgui.slider_float("Scaling Factor", self.scaling_factor["point_view"], 1e-2, 1)
                _, self.draw_poses["point_view"] = imgui.checkbox("Draw Poses", self.draw_poses["point_view"])
                _, self.draw_gt_poses["point_view"] = imgui.checkbox("Draw GT Poses", self.draw_gt_poses["point_view"])
                if self.draw_poses["point_view"] or self.draw_gt_poses["point_view"]:
                    _, self.pose_sizes["point_view"] = imgui.drag_float("Pose Sizes", self.pose_sizes["point_view"], 0.01, 0, 1e8, "%.2f")

            if self.render_mode() == "Ellipsoids":
                _, self.ellipsoid_viewer.scaling_modifier = imgui.drag_float("Scaling Factor", self.ellipsoid_viewer.scaling_modifier, v_min=0, v_max=10, v_speed=0.01)
                
                _, self.ellipsoid_viewer.render_floaters = imgui.checkbox("Render Floaters", self.ellipsoid_viewer.render_floaters)
                _, self.ellipsoid_viewer.limit = imgui.drag_float("Alpha Threshold", self.ellipsoid_viewer.limit, v_min=0, v_max=1, v_speed=0.01)
            _, self.throttling = imgui.checkbox("Throttling", self.throttling)
            if self.throttling:
                _, self.max_fps = imgui.slider_int("Max FPS", self.max_fps, 2, 60)
            _, self.keep_alive = imgui.checkbox(
                "Keep viewer alive after training", self.keep_alive)
            _, self.bg_color = imgui.color_edit3("Background Color", self.bg_color)
            if self.mode is not ViewerMode.LOCAL:
                _, self.jpeg_compression = imgui.checkbox("JPEG Compression", self.jpeg_compression)
                if self.jpeg_compression:
                    _, self.jpeg_quality = imgui.slider_int("JPEG Quality", self.jpeg_quality, 1, 100)

            imgui.separator_text("Camera Settings")
            self.snap_mode.show_gui()
            self.snap_mode_id = int(self.snap_mode.value)
            if self.snap_mode.value == SnapMode.free:
                self.reset_pose = imgui.button("Reset Pose")
                imgui.same_line()
                self.snap_to_closest = imgui.button("Snap to Closest")
                self.snap_to_closest |= imgui.is_key_pressed(imgui.Key.p)
            if self.snap_mode.value == SnapMode.keyframe:
                imgui.text("Keyframe ID")
                self.prev_keyframe = imgui.button("-")
                imgui.same_line()
                _, self.keyframe_id = imgui.slider_int("##", self.keyframe_id, 0, max(0, self.num_keyframes - 1))
                imgui.same_line()
                self.next_keyframe = imgui.button("+")
            imgui.separator()
            self.point_view_camera.show_gui()
            _, self.altitude_control = imgui.checkbox("Altitude Control", self.altitude_control)
            if self.altitude_control:
                _, self.altitude_smoothing = imgui.slider_float("smoothing", self.altitude_smoothing, 0.9, 0.9999)
            self.reset_intrinsics_flag["point_view"] = imgui.button("Reset Intrinsics")
            imgui.separator_text("Hierarchy")
            _, self.render_tau = imgui.slider_float("Render Tau", self.render_tau, 0.1, 100.0)
        with imgui_ctx.begin("Point View"):
            if self.render_mode() in ["Splats", "Depth"]:
                self.point_view.show_gui()
            else:
                self.ellipsoid_viewer.show_gui()

            if imgui.is_item_hovered():
                self.point_view_camera.process_mouse_input()
            
            if imgui.is_item_focused() or imgui.is_item_hovered():
                self.point_view_camera.process_keyboard_input()

        if self.show_top_view:
            # Default placement for the floating Top View window: anchored to
            # the bottom-right corner of the main dockspace (which excludes
            # the menu and status bars). The image is drawn at the top view
            # camera's native resolution, so size the window to wrap it
            # exactly (padding and title bar included).
            style = imgui.get_style()
            dockspace_id = hello_imgui.get_runner_params().docking_params.dock_space_id_from_name("MainDockSpace")
            if dockspace_id is not None:
                dockspace = imgui.internal.dock_builder_get_node(dockspace_id)
                corner = (dockspace.pos.x + dockspace.size.x, dockspace.pos.y + dockspace.size.y)
            else:
                viewport = imgui.get_main_viewport()
                corner = (viewport.work_pos.x + viewport.work_size.x,
                          viewport.work_pos.y + viewport.work_size.y)
            margin = hello_imgui.em_size(0.5)
            imgui.set_next_window_pos(
                (corner[0] - margin, corner[1] - margin),
                imgui.Cond_.first_use_ever, (1.0, 1.0))
            imgui.set_next_window_size(
                (self.top_view_camera.res_x + 2 * style.window_padding.x,
                 self.top_view_camera.res_y + 2 * style.window_padding.y + imgui.get_frame_height()),
                imgui.Cond_.first_use_ever)
            with imgui_ctx.begin("Top View"):
                self.top_view.show_gui()

                if imgui.is_item_hovered():
                    self.top_view_camera.process_mouse_input()
                
                if imgui.is_item_focused() or imgui.is_item_hovered():
                    self.top_view_camera.process_keyboard_input()
            
        with imgui_ctx.begin("Top View Settings"):
            _, self.show_top_view = imgui.checkbox("Show Top View", self.show_top_view)
            if self.show_top_view:
                imgui.separator_text("Render Settings")
                _, self.scaling_factor["top_view"] = imgui.slider_float("Scaling Factor", self.scaling_factor["top_view"], 1e-3, 1e-1)
                _, self.draw_poses["top_view"] = imgui.checkbox("Draw Poses", self.draw_poses["top_view"])
                _, self.draw_gt_poses["top_view"] = imgui.checkbox("Draw GT Poses", self.draw_gt_poses["top_view"])
                if self.draw_poses["top_view"] or self.draw_gt_poses["top_view"]:
                    _, self.pose_sizes["top_view"] = imgui.drag_float("Pose Sizes", self.pose_sizes["top_view"], 0.01, 0, 1e8, "%.2f")

                imgui.separator_text("Camera Settings")
                self.top_view_camera.show_gui()
                self.reset_intrinsics_flag["top_view"] = imgui.button("Reset Intrinsics")

        if self.reset_pose:
            self.point_view_camera.update_pose(np.eye(4))
        
        # Throttling
        if self.throttling:
            elapsed = time.time() - self.last_show_gui_time
            if elapsed < 1 / self.max_fps:
                time.sleep(1 / self.max_fps - elapsed)

        self.last_show_gui_time = time.time()

    def server_send(self):
        to_send = {
            "num_keyframes": self.num_keyframes,
            "keyframe_id": self.keyframe_id,
            "res_x": {view: camera.res_x for view, camera in self.cameras.items()},
            "res_y": {view: camera.res_y for view, camera in self.cameras.items()},
            "fov_x": {view: camera.fov_x for view, camera in self.cameras.items()},
            "fov_y": {view: camera.fov_y for view, camera in self.cameras.items()},
        }
        if self.updated_pose is not None:
            to_send["updated_pose"] = self.updated_pose.tolist()
        return None, to_send
    
    def client_recv(self, _, text):
        self.num_keyframes = text["num_keyframes"]
        self.keyframe_id = text["keyframe_id"]
        if "snap_mode_id" in text:
            self.set_snap_mode(text["snap_mode_id"])
        self.keep_alive = text.get("keep_alive", self.keep_alive)
        if "updated_pose" in text:
            self.updated_pose = np.array(text["updated_pose"])
        for view in self.cameras:
            self.cameras[view].res_x = text["res_x"][view]
            self.cameras[view].res_y = text["res_y"][view]
            self.cameras[view].fov_x = text["fov_x"][view]
            self.cameras[view].fov_y = text["fov_y"][view]
        
    def client_send(self):
        attrs = {key: getattr(self, key) for key in GaussianViewer.atttrs_to_sync}
        return None, attrs
    
    def server_recv(self, _, text):
        for attr in GaussianViewer.atttrs_to_sync:
            if attr in text:
                setattr(self, attr, text[attr])
    
if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title="mode", dest="mode", required=True)
    local = subparsers.add_parser("local")
    local.add_argument("scene_source", help="Scene directory, hierarchy PLY, or leaf SPZ file.")
    client = subparsers.add_parser("client")
    client.add_argument("--ip", default="127.0.0.1", help="IP address of the viewer server to connect to.")
    client.add_argument("--port", type=int, default=6009, help="Port of the viewer server to connect to.")
    server = subparsers.add_parser("server")
    server.add_argument("scene_source", help="Scene directory, hierarchy PLY, or leaf SPZ file.")
    server.add_argument("--ip", default="0.0.0.0", help="Interface to listen on. 0.0.0.0 allows remote viewer clients, use 127.0.0.1 to restrict to local connections.")
    server.add_argument("--port", type=int, default=6009, help="Port to listen on.")
    args = parser.parse_args()

    match args.mode:
        case "local":
            mode = ViewerMode.LOCAL
        case "client":
            mode = ViewerMode.CLIENT
        case "server":
            mode = ViewerMode.SERVER

    if mode is ViewerMode.CLIENT:
        viewer = GaussianViewer(mode)
    else:
        viewer = GaussianViewer.from_scene(args.scene_source, mode, args)

    if args.mode in ["client", "server"]:
        viewer.run(args.ip, args.port)
    else:
        viewer.run()
