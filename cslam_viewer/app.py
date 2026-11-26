"""Viser viewer for multi-agent 3DGS C-SLAM maps."""

import argparse
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import viser
import viser.transforms as tf

from . import dropzone
from .runs import Agent, Run, align_gt, ate_rmse, discover, estimate_up, gt_pairs
from .splats import load_splats

AGENT_COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
                (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]
LOOP_INTER, LOOP_INTRA, LOOP_REJECTED = (255, 0, 200), (255, 190, 0), (160, 160, 160)
GT_ERROR_COLOR = (40, 40, 40)  # neutral, so it never matches an agent color
AXES = {"+z": (0, 0, 1), "-z": (0, 0, -1), "+y": (0, 1, 0), "-y": (0, -1, 0), "+x": (1, 0, 0), "-x": (-1, 0, 0)}
FRUSTUM_FOV, FRUSTUM_ASPECT = np.deg2rad(60.0), 4 / 3
PLAYBACK_HZ = 30
NO_AGENT = "(no trajectories loaded)"

HELP = """Drop `.ply` maps or whole run folders anywhere on this window.

**Mouse**: left drag orbit, right drag pan, scroll zoom, Ctrl+click sets the orbit center.
**Keys**: WASD / QE / arrows fly."""


@dataclass
class MapLayer:
    centers: np.ndarray    # kept for Ctrl+click picking
    opacities: np.ndarray
    radii: np.ndarray
    bounds: np.ndarray
    handle: viser.GaussianSplatHandle
    toggle: viser.GuiCheckboxHandle


@dataclass
class RunLayer:
    key: str
    run: Run
    frame: viser.FrameHandle
    folder: viser.GuiFolderHandle
    scale: float  # rough scene size, sets frustum size and orbit distance
    maps: dict[Path, MapLayer] = field(default_factory=dict)
    map_toggles: dict[Path, viser.GuiCheckboxHandle] = field(default_factory=dict)
    traj_nodes: list = field(default_factory=list)
    cam_nodes: list = field(default_factory=list)
    loop_nodes: list = field(default_factory=list)
    rejected_nodes: list = field(default_factory=list)


class Viewer:
    def __init__(self, host: str, port: int, upload_port: int, max_splats: int, min_opacity: float):
        self.server = viser.ViserServer(host=host, port=port, label="3DGS C-SLAM viewer")
        self.server.gui.configure_theme(control_width="large")
        self.runs: dict[str, RunLayer] = {}
        self.lock = threading.RLock()
        self.up = np.array([0.0, 0.0, 1.0])
        self._run_counter = 0
        self._framed_once = False

        self.agent_entries: dict[str, tuple[RunLayer, Agent]] = {}
        self.agent: tuple[RunLayer, Agent] | None = None
        self.kf_pos = 0.0
        self.playing = False
        self._syncing_slider = False

        self._build_gui(max_splats, min_opacity)
        self.server.scene.on_click(modifier="cmd/ctrl")(self._on_ctrl_click)
        self.server.on_client_connect(self._on_connect)
        self.follow_marker = self.server.scene.add_camera_frustum(
            "/follow_marker", FRUSTUM_FOV, FRUSTUM_ASPECT, scale=0.1, thickness=3.0,
            thickness_units="screen", color=(255, 255, 255), visible=False)
        dropzone.install(self.server, host, upload_port, lambda folder: self.load_paths([folder]))
        threading.Thread(target=self._playback_loop, daemon=True).start()

    # GUI ------------------------------------------------------------------------------------

    def _build_gui(self, max_splats: int, min_opacity: float) -> None:
        gui = self.server.gui
        gui.add_markdown(HELP)

        with gui.add_folder("Load"):
            self.path_input = gui.add_text("Path", initial_value="", hint="File or folder on this machine")
            gui.add_button("Open path", icon=viser.Icon.FOLDER_OPEN).on_click(self._on_open_path)
            self.max_splats = gui.add_number("Max splats / map", initial_value=max_splats, min=1_000, step=100_000,
                                             hint="Maps above this are pruned to their most visible splats")
            self.min_opacity = gui.add_slider("Min opacity", min=0.0, max=0.5, step=0.01, initial_value=min_opacity)
            gui.add_button("Reload maps with these settings").on_click(lambda _: self._reload_all_maps())

        with gui.add_folder("View"):
            gui.add_button("Gravity down", icon=viser.Icon.ARROW_DOWN).on_click(lambda e: self.align_gravity())
            self.up_choice = gui.add_dropdown("Up axis", ("auto", *AXES), initial_value="auto",
                                              hint="auto: mean camera up of all agents, snapped to a world axis")
            gui.add_button("Frame all", icon=viser.Icon.FOCUS_CENTERED).on_click(lambda e: self.frame_all())
            self.fov = gui.add_slider("FOV (deg)", min=20.0, max=120.0, step=1.0, initial_value=60.0)
            self.fov.on_update(lambda _: self._apply_fov())

        with gui.add_folder("Follow agent"):
            self.agent_choice = gui.add_dropdown("Agent", (NO_AGENT,), initial_value=NO_AGENT)
            # Only a user's choice moves the camera, not the automatic pick when runs are (un)loaded.
            self.agent_choice.on_update(lambda e: self._on_agent_selected(move_camera=e.client is not None))
            self.kf_slider = gui.add_slider("Keyframe", min=0, max=1, step=1, initial_value=0, disabled=True)
            self.kf_slider.on_update(self._on_slider)
            self.kf_info = gui.add_markdown("")
            self.play_button = gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY, disabled=True)
            self.play_button.on_click(lambda _: self._set_playing(not self.playing))
            self.speed = gui.add_slider("Speed (kf/s)", min=1.0, max=60.0, step=1.0, initial_value=10.0)
            self.follow = gui.add_checkbox("Camera follows agent", initial_value=True)
            self.follow.on_update(lambda _: self._show_pose())
            self.level = gui.add_checkbox("Level horizon", initial_value=True,
                                          hint="Off: use the agent's exact roll instead of gravity up")
            self.level.on_update(lambda _: self._show_pose())
            gui.add_button("Snap to nearest agent view", icon=viser.Icon.CAMERA).on_click(self._on_snap)

        self.layers_folder = gui.add_folder("Layers")

    def _notify(self, title: str, body: str, error: bool = False) -> None:
        for client in self.server.get_clients().values():
            client.add_notification(title, body, auto_close_seconds=6.0 if not error else 12.0,
                                    color="red" if error else None)

    def _on_connect(self, client: viser.ClientHandle) -> None:
        client.camera.fov = np.deg2rad(self.fov.value)
        framed = threading.Event()

        # The camera can only be moved once the browser has reported its initial state.
        @client.camera.on_update
        def _(_) -> None:
            if self.runs and not framed.is_set():
                framed.set()
                self.frame_all([client])

    def _on_open_path(self, _) -> None:
        raw = self.path_input.value.strip().strip("'\"")
        if raw:
            threading.Thread(target=self.load_paths, args=([Path(raw)],), daemon=True).start()

    # Loading --------------------------------------------------------------------------------

    def load_paths(self, paths: list[Path]) -> None:
        try:
            missing = [p for p in paths if not p.expanduser().exists()]
            if missing:
                raise FileNotFoundError(f"Not found: {missing[0]}")
            runs = discover(paths)
        except Exception as e:  # noqa: BLE001 - shown to the user instead of crashing the thread
            self._notify("Could not load", str(e), error=True)
            return
        if not runs:
            self._notify("Nothing to load", "No .ply maps or run folders found.", error=True)
        for run in runs:
            self._add_run(run)

    def _add_run(self, run: Run) -> None:
        with self.lock:
            self._run_counter += 1
            key = f"r{self._run_counter}"
        positions = [a.c2w[:, :3, 3] for a in run.agents]
        scale = float(np.linalg.norm(np.ptp(np.concatenate(positions), axis=0))) if positions else 1.0

        frame = self.server.scene.add_frame(f"/runs/{key}", show_axes=False)
        with self.layers_folder:
            folder = self.server.gui.add_folder(run.name)
        layer = RunLayer(key, run, frame, folder, max(scale, 0.5))
        with self.lock:
            self.runs[key] = layer

        with folder:
            for path in run.map_paths:
                toggle = self.server.gui.add_checkbox(path.stem, initial_value=True, disabled=True, hint="loading...")
                layer.map_toggles[path] = toggle
                self._bind_map_toggle(layer, path, toggle)
            self._add_trajectories(layer)
            self.server.gui.add_button("Remove", icon=viser.Icon.TRASH, color="red").on_click(
                lambda _: self._remove_run(key))
        self._refresh_agents()

        for path in run.map_paths:
            self._load_map(layer, path)

        if not self._framed_once and key in self.runs:
            self._framed_once = True
            self.align_gravity(quiet=True)
            self.frame_all()

    def _load_map(self, layer: RunLayer, path: Path) -> None:
        toggle = layer.map_toggles[path]
        notes = [c.add_notification("Loading map", path.name, loading=True, with_close_button=False)
                 for c in self.server.get_clients().values()]
        try:
            s = load_splats(path, int(self.max_splats.value), float(self.min_opacity.value))
        except Exception as e:  # noqa: BLE001 - shown to the user instead of crashing the thread
            toggle.hint = f"failed: {e}"
            self._notify(f"Could not load {path.name}", str(e), error=True)
            return
        finally:
            for n in notes:
                n.remove()
        if layer.key not in self.runs:
            return  # removed while loading
        handle = self.server.scene.add_gaussian_splats(
            f"/runs/{layer.key}/maps/{path.stem}", s.centers, s.covariances, s.rgbs, s.opacities,
            visible=toggle.value)
        layer.maps[path] = MapLayer(s.centers, s.opacities, s.radii, s.bounds, handle, toggle)
        toggle.hint = f"{len(s.centers) / 1e6:.2f}M of {s.total / 1e6:.2f}M splats shown"
        toggle.disabled = False

    @staticmethod
    def _bind_map_toggle(layer: RunLayer, path: Path, toggle: viser.GuiCheckboxHandle) -> None:
        def update(_):
            if path in layer.maps:
                layer.maps[path].handle.visible = toggle.value
        toggle.on_update(update)

    def _reload_all_maps(self) -> None:
        def work():
            for layer in list(self.runs.values()):
                for path in list(layer.maps):
                    layer.maps.pop(path).handle.remove()
                    layer.map_toggles[path].disabled = True
                    self._load_map(layer, path)
        threading.Thread(target=work, daemon=True).start()

    def _add_trajectories(self, layer: RunLayer) -> None:
        run, scene, base = layer.run, self.server.scene, f"/runs/{layer.key}"
        size = float(np.clip(0.01 * layer.scale, 0.02, 1.5))
        for a in run.agents:
            color = AGENT_COLORS[a.id % len(AGENT_COLORS)]
            pos = a.c2w[:, :3, 3]
            if len(pos) >= 2:
                layer.traj_nodes.append(scene.add_line_segments(
                    f"{base}/traj/{a.id}", np.stack([pos[:-1], pos[1:]], axis=1), color,
                    thickness=3.0, thickness_units="screen"))
            layer.cam_nodes.append(scene.add_line_segments(
                f"{base}/cams/{a.id}", frustum_segments(a.c2w, size), color,
                thickness=1.0, thickness_units="screen", visible=False))

        if run.loops is not None:
            lp = run.loops
            segs = np.stack([lp.src_pos, lp.tgt_pos], axis=1)
            colors = np.where(lp.intra[:, None], LOOP_INTRA, LOOP_INTER).astype(np.uint8)
            if lp.accepted.any():
                layer.loop_nodes.append(scene.add_line_segments(
                    f"{base}/loops/accepted", segs[lp.accepted],
                    np.repeat(colors[lp.accepted][:, None], 2, axis=1), thickness=2.5, thickness_units="screen"))
            if (~lp.accepted).any():
                layer.rejected_nodes.append(scene.add_line_segments(
                    f"{base}/loops/rejected", segs[~lp.accepted], LOOP_REJECTED,
                    thickness=1.0, thickness_units="screen", visible=False))

        gui = self.server.gui
        if run.agents:
            self._node_toggle(gui.add_checkbox("Trajectories", initial_value=True), layer.traj_nodes)
            self._node_toggle(gui.add_checkbox("Keyframe cameras", initial_value=False), layer.cam_nodes)
        if run.loops is not None:
            n_acc = int(run.loops.accepted.sum())
            n_inter = int((run.loops.accepted & ~run.loops.intra).sum())
            n_rej = len(run.loops.accepted) - n_acc
            if n_acc:
                self._node_toggle(gui.add_checkbox(
                    f"Loops ({n_acc})", initial_value=True,
                    hint=f"{n_inter} inter-agent (magenta), {n_acc - n_inter} intra-agent (yellow)"),
                    layer.loop_nodes)
            if n_rej:
                self._node_toggle(gui.add_checkbox(f"Rejected loops ({n_rej})", initial_value=False, hint="grey"),
                                  layer.rejected_nodes)
        self._add_ground_truth(layer)

    def _add_ground_truth(self, layer: RunLayer) -> None:
        agents = [a for a in layer.run.agents if a.gt_kf_c2w is not None]
        if not agents:
            return
        scene, gui, base = self.server.scene, self.server.gui, f"/runs/{layer.key}"
        # The GT lines hang under their own frame, so aligning them is just that frame's transform.
        gt_frame = scene.add_frame(f"{base}/gt", show_axes=False)
        for a in agents:
            poses = a.gt_c2w if a.gt_c2w is not None else a.gt_kf_c2w
            pos = poses[np.isfinite(poses).all(axis=(1, 2)), :3, 3]
            if len(pos) >= 2:
                scene.add_line_segments(f"{base}/gt/{a.id}", np.stack([pos[:-1], pos[1:]], axis=1),
                                        lighten(AGENT_COLORS[a.id % len(AGENT_COLORS)]),
                                        thickness=2.0, thickness_units="screen")
        alignment = align_gt(agents)

        show = gui.add_checkbox("Ground truth", initial_value=True, hint="Lighter shade of each agent's color")
        errors = gui.add_checkbox("GT error lines", initial_value=False,
                                  hint="Dark line from each estimated keyframe to its ground truth")
        align = gui.add_checkbox("Align GT to estimate", initial_value=False,
                                 hint="One rigid fit over all agents (Umeyama), as for aligned ATE")
        ate_text = gui.add_markdown("")

        def update(_=None) -> None:
            T = alignment if align.value else np.eye(4)
            gt_frame.wxyz = tf.SO3.from_matrix(T[:3, :3]).wxyz
            gt_frame.position = T[:3, 3]
            gt_frame.visible = show.value
            segments = [np.stack([est, gt @ T[:3, :3].T + T[:3, 3]], axis=1) for est, gt in map(gt_pairs, agents)]
            scene.add_line_segments(f"{base}/gt_errors", np.concatenate(segments), GT_ERROR_COLOR,
                                    thickness=1.0, thickness_units="screen", visible=errors.value)
            label = "ATE RMSE, aligned" if align.value else "ATE RMSE"
            ate_text.content = f"{label}: " + ", ".join(
                f"agent {a.id} {format_length(ate_rmse(a, T))}" for a in agents)

        for checkbox in (show, errors, align):
            checkbox.on_update(update)
        update()

    @staticmethod
    def _node_toggle(checkbox: viser.GuiCheckboxHandle, nodes: list) -> None:
        def update(_):
            for n in nodes:
                n.visible = checkbox.value
        checkbox.on_update(update)

    def _remove_run(self, key: str) -> None:
        with self.lock:
            layer = self.runs.pop(key, None)
        if layer is None:
            return
        layer.frame.remove()
        layer.folder.remove()
        self._refresh_agents()

    # Follow agent ---------------------------------------------------------------------------

    def _refresh_agents(self) -> None:
        entries = {}
        for layer in list(self.runs.values()):
            for a in layer.run.agents:
                label = f"{layer.run.name} / agent {a.id}"
                if label in entries:
                    label += f" ({layer.key})"
                entries[label] = (layer, a)
        self.agent_entries = entries
        # The options setter already fires the dropdown callback when it has to change the value.
        self.agent_choice.options = tuple(entries) or (NO_AGENT,)
        if entries.get(self.agent_choice.value) is not self.agent:
            self._on_agent_selected()

    def _on_agent_selected(self, move_camera: bool = False) -> None:
        self.agent = self.agent_entries.get(self.agent_choice.value)
        if self.agent is None:
            self._set_playing(False)
            self.kf_slider.disabled = self.play_button.disabled = True
            self.follow_marker.visible = False
            self.kf_info.content = ""
            return
        layer, agent = self.agent
        last = max(len(agent.c2w) - 1, 1)
        self._syncing_slider = True
        self.kf_slider.value = min(self.kf_slider.value, last)
        self.kf_slider.max = last
        self._syncing_slider = False
        self.kf_slider.disabled = self.play_button.disabled = False
        self.kf_pos = float(min(self.kf_pos, len(agent.c2w) - 1))
        self.follow_marker.color = AGENT_COLORS[agent.id % len(AGENT_COLORS)]
        self.follow_marker.scale = float(np.clip(0.03 * layer.scale, 0.05, 3.0))
        self._show_pose(move_camera)

    def _on_slider(self, _) -> None:
        if not self._syncing_slider and self.agent is not None:
            self.kf_pos = float(self.kf_slider.value)
            self._show_pose()

    def _set_playing(self, playing: bool) -> None:
        if playing and self.agent is not None and self.kf_pos >= len(self.agent[1].c2w) - 1:
            self.kf_pos = 0.0
        self.playing = playing and self.agent is not None
        self.play_button.label = "Pause" if self.playing else "Play"
        self.play_button.icon = viser.Icon.PLAYER_PAUSE if self.playing else viser.Icon.PLAYER_PLAY

    def _playback_loop(self) -> None:
        last = time.perf_counter()
        while True:
            time.sleep(1.0 / PLAYBACK_HZ)
            now = time.perf_counter()
            dt, last = now - last, now
            if not self.playing or self.agent is None:
                continue
            end = len(self.agent[1].c2w) - 1
            self.kf_pos = min(self.kf_pos + self.speed.value * dt, end)
            self._show_pose()
            self._syncing_slider = True
            self.kf_slider.value = int(self.kf_pos)
            self._syncing_slider = False
            if self.kf_pos >= end:
                self._set_playing(False)

    def _show_pose(self, move_camera: bool = True) -> None:
        if self.agent is None:
            return
        layer, agent = self.agent
        R, t = interpolate_pose(agent.c2w, self.kf_pos)
        self.follow_marker.wxyz = tf.SO3.from_matrix(R).wxyz
        self.follow_marker.position = t
        self.follow_marker.visible = not (self.follow.value and move_camera)
        k = int(self.kf_pos)
        self.kf_info.content = f"Keyframe {k + 1} / {len(agent.c2w)} · frame {int(agent.kf_ids[k])}"
        if self.follow.value and move_camera:
            orbit = float(np.clip(0.05 * layer.scale, 0.2, 5.0))
            for client in self.server.get_clients().values():
                self._set_camera(client, R, t, orbit)

    def _set_camera(self, client: viser.ClientHandle, R: np.ndarray, t: np.ndarray, orbit: float) -> None:
        up = self.up if self.level.value else -R[:, 1]
        try:
            with client.atomic():
                client.camera.up_direction = up
                client.camera.position = t
                client.camera.look_at = t + R[:, 2] * orbit
        except (AssertionError, ValueError):
            pass  # client has not reported its camera yet

    def _on_snap(self, event: viser.GuiEvent) -> None:
        if event.client is None or not self.agent_entries:
            return
        cam = event.client.camera
        p, fwd = cam.position, tf.SO3(cam.wxyz).as_matrix()[:, 2]
        best = None
        for label, (layer, a) in self.agent_entries.items():
            pos, dirs = a.c2w[:, :3, 3], a.c2w[:, :3, 2]
            # Position dominates; view direction breaks ties between nearby keyframes.
            cost = np.linalg.norm(pos - p, axis=1) + 0.2 * layer.scale * (1.0 - dirs @ fwd)
            k = int(np.argmin(cost))
            if best is None or cost[k] < best[0]:
                best = (cost[k], label, k)
        _, label, k = best
        self._set_playing(False)
        self.kf_pos = float(k)
        self.agent_choice.value = label
        self._syncing_slider = True
        self.kf_slider.value = k
        self._syncing_slider = False
        self.follow.value = True
        self._show_pose()

    # View -----------------------------------------------------------------------------------

    def align_gravity(self, quiet: bool = False) -> None:
        if self.up_choice.value == "auto":
            agents = [a for layer in list(self.runs.values()) for a in layer.run.agents]
            up = estimate_up(agents)
            source = f"mean camera up of {len(agents)} agents"
            if up is None:
                up, source = np.array([0.0, 0.0, 1.0]), "no trajectories, using +Z"
        else:
            up, source = np.array(AXES[self.up_choice.value], float), "manual"
        self.up = up / np.linalg.norm(up)
        self.server.scene.set_up_direction(tuple(self.up))
        for client in self.server.get_clients().values():
            try:
                client.camera.up_direction = self.up
            except (AssertionError, ValueError):
                pass  # client has not reported its camera yet
        if not quiet:
            self._notify("Gravity down", f"Up = {np.round(self.up, 3).tolist()} ({source})")

    def frame_all(self, clients: list[viser.ClientHandle] | None = None) -> None:
        boxes = [m.bounds for layer in list(self.runs.values()) for m in layer.maps.values() if m.handle.visible]
        if not boxes:
            boxes = [np.stack([a.c2w[:, :3, 3].min(0), a.c2w[:, :3, 3].max(0)])
                     for layer in list(self.runs.values()) for a in layer.run.agents]
        if not boxes:
            return
        lo = np.min([b[0] for b in boxes], axis=0)
        hi = np.max([b[1] for b in boxes], axis=0)
        center, diag = (lo + hi) / 2, max(float(np.linalg.norm(hi - lo)), 0.1)
        side = np.cross(self.up, [1.0, 0.0, 0.0] if abs(self.up[0]) < 0.9 else [0.0, 1.0, 0.0])
        offset = side / np.linalg.norm(side) + 0.7 * self.up
        position = center + 0.8 * diag * offset / np.linalg.norm(offset)
        for client in clients if clients is not None else self.server.get_clients().values():
            try:
                with client.atomic():
                    client.camera.far = max(1000.0, 4.0 * diag)
                    client.camera.up_direction = self.up
                    client.camera.position = position
                    client.camera.look_at = center
            except (AssertionError, ValueError):
                pass

    def _apply_fov(self) -> None:
        for client in self.server.get_clients().values():
            client.camera.fov = np.deg2rad(self.fov.value)

    def _on_ctrl_click(self, event: viser.SceneClickEvent) -> None:
        origin = np.asarray(event.ray_origin)
        direction = np.asarray(event.ray_direction)
        direction /= np.linalg.norm(direction)
        hits = [h for layer in list(self.runs.values()) for m in layer.maps.values()
                if m.handle.visible and (h := pick_depth(m.centers, m.opacities, m.radii, origin, direction)) is not None]
        if not hits:
            self._notify("No splat under cursor", "Ctrl+click on the map to set the orbit center.")
            return
        event.client.camera.look_at = origin + direction * min(hits)


def lighten(color: tuple[int, int, int], amount: float = 0.55) -> tuple[int, int, int]:
    return tuple(int(c + (255 - c) * amount) for c in color)


def format_length(meters: float) -> str:
    if meters < 0.01:
        return f"{meters * 1000:.1f} mm"
    return f"{meters * 100:.1f} cm" if meters < 1.0 else f"{meters:.2f} m"


def frustum_segments(c2w: np.ndarray, size: float) -> np.ndarray:
    """(K, 4, 4) poses to (K * 8, 2, 3) wireframe segments in the OpenCV camera convention."""
    h = np.tan(FRUSTUM_FOV / 2) * size
    w = h * FRUSTUM_ASPECT
    corners_cam = np.array([[-w, -h, size], [w, -h, size], [w, h, size], [-w, h, size]])
    corners = np.einsum("kij,cj->kci", c2w[:, :3, :3], corners_cam) + c2w[:, None, :3, 3]
    apex = np.broadcast_to(c2w[:, None, :3, 3], corners.shape)
    spokes = np.stack([apex, corners], axis=2)
    rim = np.stack([corners, np.roll(corners, -1, axis=1)], axis=2)
    return np.concatenate([spokes, rim], axis=1).reshape(-1, 2, 3)


def interpolate_pose(c2w: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    """Pose at fractional keyframe index s: slerp on rotation, lerp on translation."""
    i = int(np.clip(np.floor(s), 0, len(c2w) - 1))
    a = s - i
    if a < 1e-6 or i == len(c2w) - 1:
        return c2w[i, :3, :3], c2w[i, :3, 3]
    R0, R1 = c2w[i, :3, :3], c2w[i + 1, :3, :3]
    delta = tf.SO3.from_matrix(R0.T @ R1).log()
    R = R0 @ tf.SO3.exp(delta * a).as_matrix()
    return R, (1 - a) * c2w[i, :3, 3] + a * c2w[i + 1, :3, 3]


def pick_depth(centers: np.ndarray, opacities: np.ndarray, radii: np.ndarray, origin: np.ndarray,
               direction: np.ndarray, cone_deg: float = 0.3) -> float | None:
    """Depth along the ray of the first fairly opaque splat the ray passes through."""
    v = centers - origin.astype(np.float32)
    depth = v @ direction.astype(np.float32)
    perp2 = np.einsum("ij,ij->i", v, v) - depth ** 2
    # A splat is hit when the ray passes within its own extent, or a small cone for tiny splats.
    radius = np.maximum(radii, depth * np.tan(np.deg2rad(cone_deg)))
    candidates = depth[(depth > 0) & (perp2 < radius ** 2) & (opacities[:, 0] > 0.2)]
    if not len(candidates):
        return None
    # A few hits in, so a single floater in front of the surface does not win.
    k = min(3, len(candidates) - 1)
    return float(np.partition(candidates, k)[k])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Run folders, folders of runs, or .ply files")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--upload-port", type=int, default=None, help="Defaults to --port + 1")
    parser.add_argument("--max-splats", type=int, default=3_000_000)
    parser.add_argument("--min-opacity", type=float, default=0.05)
    args = parser.parse_args()
    # Turn SIGTERM into a normal exit so the upload temp folder is cleaned up.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    viewer = Viewer(args.host, args.port, args.upload_port or args.port + 1, args.max_splats, args.min_opacity)
    if args.paths:
        threading.Thread(target=viewer.load_paths, args=(args.paths,), daemon=True).start()
    try:
        viewer.server.sleep_forever()
    except KeyboardInterrupt:
        pass  # Ctrl+C: exit quietly; atexit still removes the upload temp folder.


if __name__ == "__main__":
    main()
