"""Auto-record common library for the Global Humanoid Robot Challenge 2026.

Separation of concerns (SoC / SRP):
    - SimRuntime          : boots Isaac Sim + scene + robot + IK + planner.
    - LeRobotRecorder     : wraps LeRobotDataset (add_frame / save_episode).
    - GripperHelper       : direct finger control (baseline lacks this API).
    - SinglePickPlaceFSM  : single-arm pick-and-place state machine.
    - run_episode         : one-episode loop, decoupled from task-specific logic.

Each auto_record_taskN.py only needs to provide:
    - task_yaml, repo_id, num_episodes
    - place_target_fn(part_idx, side, grasp_pose_base) -> place_pose_base
    - (optional) randomize_fn(runtime, ep_idx) for domain randomization
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

# ---- Sim / lerobot constants -------------------------------------------
PHYSICS_DT = 1.0 / 200.0       # matches WalkerS2SimRobotConfig
DEFAULT_FPS = 30
STATE_DIM = 18                 # 7 left arm + 2 left finger + 7 right arm + 2 right finger
ACTION_DIM = 18

# Reference values from IsaacSimRobotInterface.
GRIPPER_OPEN = -0.0215
GRIPPER_CLOSE = 0.011

CAMERAS = ("head_left", "head_right", "wrist_left", "wrist_right")
CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640


# ------------------------------------------------------------------------
# Sim runtime
# ------------------------------------------------------------------------
@dataclass
class SimRuntimeConfig:
    task_yaml: str
    headless: bool = True
    fps: int = DEFAULT_FPS
    settle_time: float = 1.0
    seed: int = 0


class SimRuntime:
    """Boot and tear down Isaac Sim + Ubtech scene.

    Kept in a dedicated class so each task script is freed from knowing the
    details of SimulationApp / World / SceneBuilder (SRP).
    """

    def __init__(self, cfg: SimRuntimeConfig):
        self.cfg = cfg
        self.kit = None
        self.world = None
        self.scene = None
        self.robot = None
        self.ik_solver = None
        self.coord_transform = None
        self.planner = None
        self.task_cfg = None
        self.phys_per_frame = max(1, round((1.0 / cfg.fps) / PHYSICS_DT))

    # ---- public lifecycle ----------------------------------------------
    def boot(self):
        """Open SimulationApp and build stage / scene / robot / IK / planner."""
        from isaacsim import SimulationApp

        launch_cfg = {"headless": self.cfg.headless, "width": 1280, "height": 720}
        self.kit = SimulationApp(launch_config=launch_cfg)

        # Imports must come AFTER SimulationApp is created.
        import omni
        import omni.replicator.core as rep
        from isaacsim.core.api import World

        from source.config_loader import load_config, apply_scatter_config
        from source.SceneBuilder import SceneBuilder
        from source.RobotArticulation import RobotArticulation
        from source.DataLogger import DataLogger
        from source.coordinate_utils_v2 import CoordinateTransform
        from source.grasp_planner import GraspPlanner

        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        self.task_cfg = load_config(self.cfg.task_yaml)
        grasp_cfg = self.task_cfg.get("grasp", {})

        # Stage + World ------------------------------------------------
        omni.usd.get_context().open_stage(
            os.path.join(self.task_cfg["root_path"], self.task_cfg["scene_usd"])
        )
        rendering_dt = 1.0 / self.cfg.fps
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=PHYSICS_DT,
            rendering_dt=rendering_dt,
        )
        self.world.initialize_physics()

        # DataLogger is required by SceneBuilder.__init__ but we record
        # via LeRobotDataset, so disable both CSV and HDF5 outputs.
        data_logger = DataLogger(
            enabled=False,
            csv_path=os.path.join(os.path.dirname(__file__), "_dummy_poses.csv"),
            camera_enabled=False,
            camera_hdf5_path=os.path.join(os.path.dirname(__file__), "_dummy.hdf5"),
        )

        self.scene = SceneBuilder(self.task_cfg, data_logger=data_logger)
        apply_scatter_config(self.task_cfg)
        self.scene.build_all()
        rep.orchestrator.step()

        # Settle physics ----------------------------------------------
        self.world.play()
        settle_steps = int(self.cfg.settle_time / self.world.get_physics_dt())
        for _ in range(settle_steps):
            self.world.step(render=False)

        # Robot --------------------------------------------------------
        self.world.pause()
        self.scene.build_robot()
        self.robot = RobotArticulation(prim_path="/Root/Ref_Xform/Ref", name="walkerS2")
        self.robot.initialize()
        self.world.play()
        for _ in range(10):
            self.world.step(render=False)

        # IK + coordinate transform -----------------------------------
        urdf_path = os.path.join(self.task_cfg["root_path"], "s2.urdf")
        self.robot.initialize_ik(urdf_path)
        js = self.robot.get_joint_states()
        if js is not None:
            self.robot.ik_solver.sync_joint_positions(js["names"], js["positions"][0])

        compensation_matrix = np.array([
            [9.99999e-01, -1.11400e-03,  1.16200e-03, -9.64000e-04],
            [-2.00000e-05,  7.13609e-01,  7.00544e-01, -9.59927e-01],
            [-1.61000e-03, -7.00544e-01,  7.13608e-01,  6.56540e-01],
            [0.0,           0.0,          0.0,          1.0],
        ], dtype=np.float64)
        self.coord_transform = CoordinateTransform.from_torso_link(
            ik_solver=self.robot.ik_solver,
            compensation_matrix=compensation_matrix,
        )

        # Planner ------------------------------------------------------
        self.planner = GraspPlanner(grasp_cfg, self.robot, self.coord_transform)

    def shutdown(self):
        if self.kit is not None:
            try:
                self.kit.close()
            except Exception:
                pass

    # ---- scene control --------------------------------------------------
    def reset_episode(self):
        """Re-scatter parts for a new episode (scene already built)."""
        self.world.pause()
        self.scene.scatter_after_reset()
        self.world.play()
        settle_steps = int(self.cfg.settle_time / self.world.get_physics_dt())
        for _ in range(settle_steps):
            self.world.step(render=False)

    def step_one_frame(self):
        """Advance phys_per_frame-1 steps without render, then 1 step with render."""
        for _ in range(self.phys_per_frame - 1):
            self.world.step(render=False)
        self.world.step(render=True)


# ------------------------------------------------------------------------
# Gripper helper (RobotArticulation lacks a finger API)
# ------------------------------------------------------------------------
class GripperHelper:
    """Drive the L/R fingers via low-level articulation actions."""

    def __init__(self, articulation):
        import torch  # local import: only available inside Isaac Sim env
        self._torch = torch
        self.arts = articulation
        self._left_idx = [
            articulation.get_dof_index("L_finger1_joint"),
            articulation.get_dof_index("L_finger2_joint"),
        ]
        self._right_idx = [
            articulation.get_dof_index("R_finger1_joint"),
            articulation.get_dof_index("R_finger2_joint"),
        ]
        self._state = {"left": GRIPPER_OPEN, "right": GRIPPER_OPEN}

    def set(self, side: str, width: float):
        from isaacsim.core.utils.types import ArticulationActions

        idx = self._left_idx if side == "left" else self._right_idx
        self._state[side] = width
        self.arts.apply_action(ArticulationActions(
            joint_positions=self._torch.tensor([[width, width]], dtype=self._torch.float32),
            joint_indices=self._torch.tensor(idx, dtype=self._torch.int32),
        ))

    def open(self, side: str):
        self.set(side, GRIPPER_OPEN)

    def close(self, side: str):
        self.set(side, GRIPPER_CLOSE)

    def get_state(self) -> dict:
        return dict(self._state)


# ------------------------------------------------------------------------
# Pick-place FSM
# ------------------------------------------------------------------------
PICK_PLACE_PHASES = (
    "APPROACH", "DESCEND", "GRASP", "LIFT",
    "TRANSPORT", "PLACE", "RELEASE", "RETREAT", "DONE",
)


@dataclass
class FSMTimings:
    """Frame budget per phase (assumes fps=30; total ~9 s/part default)."""
    approach: int = 30
    descend: int = 25
    grasp: int = 20
    lift: int = 25
    transport: int = 35
    place: int = 25
    release: int = 15
    retreat: int = 25


class SinglePickPlaceFSM:
    """Pick & place one part with one arm.

    `step()` returns (left_target, right_target, side, gripper_cmd, done).
    The idle arm is held at its `rest_*` pose.
    """

    APPROACH_HEIGHT = 0.10
    LIFT_HEIGHT = 0.17
    PLACE_APPROACH = 0.10
    PLACE_DESCEND = 0.03

    def __init__(
        self,
        grasp_pose: np.ndarray,        # 6-D xyzrpy in base frame, tcp-offset already applied
        place_pose: np.ndarray,        # 6-D xyzrpy in base frame, tcp-offset already applied
        side: str,                     # 'left' | 'right'
        rest_left: np.ndarray,
        rest_right: np.ndarray,
        timings: FSMTimings | None = None,
    ):
        self.grasp = np.asarray(grasp_pose, dtype=float).copy()
        self.place = np.asarray(place_pose, dtype=float).copy()
        self.side = side
        self.rest_left = np.asarray(rest_left, dtype=float).copy()
        self.rest_right = np.asarray(rest_right, dtype=float).copy()
        self.t = timings or FSMTimings()
        self.phase = "APPROACH"
        self._frames_in_phase = 0

    @property
    def done(self) -> bool:
        return self.phase == "DONE"

    def _budget(self) -> int:
        return getattr(self.t, self.phase.lower(), 30)

    def _advance(self):
        order = list(PICK_PLACE_PHASES)
        i = order.index(self.phase)
        self.phase = order[i + 1] if i + 1 < len(order) else "DONE"
        self._frames_in_phase = 0

    def _active_target(self) -> tuple[np.ndarray, str]:
        g = self.grasp.copy()
        p = self.place.copy()

        if self.phase == "APPROACH":
            tgt = g.copy(); tgt[2] += self.APPROACH_HEIGHT
            return tgt, "open"
        if self.phase == "DESCEND":
            return g, "open"
        if self.phase == "GRASP":
            return g, "close"
        if self.phase == "LIFT":
            tgt = g.copy(); tgt[2] += self.LIFT_HEIGHT
            return tgt, "close"
        if self.phase == "TRANSPORT":
            tgt = p.copy(); tgt[2] += self.PLACE_APPROACH
            return tgt, "close"
        if self.phase == "PLACE":
            tgt = p.copy(); tgt[2] += self.PLACE_DESCEND
            return tgt, "close"
        if self.phase == "RELEASE":
            tgt = p.copy(); tgt[2] += self.PLACE_DESCEND
            return tgt, "open"
        if self.phase == "RETREAT":
            tgt = p.copy(); tgt[2] += self.LIFT_HEIGHT
            return tgt, "open"
        rest = self.rest_left if self.side == "left" else self.rest_right
        return rest, "open"

    def step(self):
        active_target, gripper = self._active_target()

        if self.side == "left":
            left_target, right_target = active_target, self.rest_right
        else:
            left_target, right_target = self.rest_left, active_target

        self._frames_in_phase += 1
        if self._frames_in_phase >= self._budget() and not self.done:
            self._advance()

        return left_target, right_target, self.side, gripper, self.done


# ------------------------------------------------------------------------
# LeRobot dataset wrapper
# ------------------------------------------------------------------------
def make_features() -> dict:
    """18-D state/action + 4 RGB camera videos."""
    feats: dict = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": [f"q_{i}" for i in range(STATE_DIM)],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": [f"a_{i}" for i in range(ACTION_DIM)],
        },
    }
    for cam in CAMERAS:
        feats[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
            "names": ["height", "width", "channel"],
        }
    return feats


class LeRobotRecorder:
    """Thin wrapper over LeRobotDataset.create / add_frame / save_episode."""

    def __init__(self, repo_id: str, root: Path, fps: int = DEFAULT_FPS,
                 image_writer_processes: int = 0,
                 image_writer_threads: int = 4):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=str(root),
            features=make_features(),
            use_videos=True,
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
        )

    def add(self, state18: np.ndarray, action18: np.ndarray,
            images: dict[str, np.ndarray], task_name: str):
        frame = {
            "observation.state": state18.astype(np.float32),
            "action": action18.astype(np.float32),
            "task": task_name,
        }
        for cam, img in images.items():
            frame[f"observation.images.{cam}"] = img
        self.dataset.add_frame(frame)

    def save_episode(self):
        self.dataset.save_episode()


# ------------------------------------------------------------------------
# Observation / action helpers
# ------------------------------------------------------------------------
ARM_JOINTS_18 = (
    "L_shoulder_pitch_joint", "L_shoulder_roll_joint", "L_shoulder_yaw_joint",
    "L_elbow_pitch_joint", "L_elbow_yaw_joint",
    "L_wrist_pitch_joint", "L_wrist_roll_joint",
    "L_finger1_joint", "L_finger2_joint",
    "R_shoulder_pitch_joint", "R_shoulder_roll_joint", "R_shoulder_yaw_joint",
    "R_elbow_pitch_joint", "R_elbow_yaw_joint",
    "R_wrist_pitch_joint", "R_wrist_roll_joint",
    "R_finger1_joint", "R_finger2_joint",
)


def gather_state_18(robot) -> np.ndarray:
    js = robot.get_joint_states()
    if js is None:
        return np.zeros(STATE_DIM, dtype=np.float32)
    names = js["names"]
    pos = js["positions"][0]
    out = np.zeros(STATE_DIM, dtype=np.float32)
    name_to_pos = {n: float(p) for n, p in zip(names, pos)}
    for i, jn in enumerate(ARM_JOINTS_18):
        out[i] = name_to_pos.get(jn, 0.0)
    return out


def gather_camera_images(robot) -> dict[str, np.ndarray]:
    out = {}
    for cam in CAMERAS:
        img = robot.get_camera_rgb(cam)
        if img is None:
            img = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        else:
            img = np.asarray(img)
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            if img.shape[-1] == 4:
                img = img[..., :3]
        out[cam] = img
    return out


# ------------------------------------------------------------------------
# Episode runner
# ------------------------------------------------------------------------
PlaceTargetFn = Callable[[int, str, np.ndarray], np.ndarray]
RandomizeFn = Callable[["SimRuntime", int], None]


@dataclass
class EpisodeSpec:
    task_name: str
    num_parts: int
    place_target_fn: PlaceTargetFn
    randomize_fn: RandomizeFn | None = None
    timings: FSMTimings = field(default_factory=FSMTimings)
    max_extra_frames: int = 60
    target_indices: Iterable[int] | None = None


def run_episode(runtime: SimRuntime, recorder: LeRobotRecorder, spec: EpisodeSpec,
                ep_idx: int) -> bool:
    """Run one pick-and-place episode and persist it to the recorder."""
    runtime.reset_episode()

    if spec.randomize_fn is not None:
        spec.randomize_fn(runtime, ep_idx)

    gripper = GripperHelper(runtime.robot._articulation)

    indices = list(spec.target_indices) if spec.target_indices is not None \
        else list(range(spec.num_parts))

    for part_idx in indices:
        part_poses = runtime.scene.get_parts_world_poses()
        if part_idx >= len(part_poses):
            print(f"[ep {ep_idx}] part_idx {part_idx} >= {len(part_poses)} parts; skip")
            continue

        runtime.planner.target_index = part_idx
        runtime.planner.compute_grasp_target(part_poses)
        if runtime.planner.active_grasp is None:
            print(f"[ep {ep_idx}] planner failed for part {part_idx}; skip")
            continue
        grasp_pose = runtime.planner.active_grasp.copy()
        side = runtime.planner.grasp_arm
        place_pose = spec.place_target_fn(part_idx, side, grasp_pose)

        fsm = SinglePickPlaceFSM(
            grasp_pose=grasp_pose,
            place_pose=place_pose,
            side=side,
            rest_left=runtime.planner.left_init,
            rest_right=runtime.planner.right_init,
            timings=spec.timings,
        )

        gripper.open(side)

        while not fsm.done:
            obs_state = gather_state_18(runtime.robot)
            obs_imgs = gather_camera_images(runtime.robot)

            left_t, right_t, fsm_side, grip_cmd, _ = fsm.step()
            runtime.robot.control_dual_arm_ik(
                step_size=PHYSICS_DT,
                left_target_xyzrpy=left_t,
                right_target_xyzrpy=right_t,
                rot_weight=runtime.planner.ik_rot_weight,
            )
            if grip_cmd == "open":
                gripper.open(fsm_side)
            else:
                gripper.close(fsm_side)

            runtime.step_one_frame()

            # Action recorded for this frame is the achieved next state, which
            # is the pattern used by ACT/Pi0 on the lerobot side.
            action = gather_state_18(runtime.robot)
            recorder.add(obs_state, action, obs_imgs, spec.task_name)

    extra = spec.max_extra_frames
    while extra > 0:
        obs_state = gather_state_18(runtime.robot)
        obs_imgs = gather_camera_images(runtime.robot)
        runtime.robot.control_dual_arm_ik(
            step_size=PHYSICS_DT,
            left_target_xyzrpy=runtime.planner.left_init,
            right_target_xyzrpy=runtime.planner.right_init,
            rot_weight=runtime.planner.ik_rot_weight,
        )
        runtime.step_one_frame()
        action = gather_state_18(runtime.robot)
        recorder.add(obs_state, action, obs_imgs, spec.task_name)
        extra -= 1

    recorder.save_episode()
    return True


# ------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------
def parse_cli(default_repo_id: str, default_root: str) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-episodes", type=int, default=1500)
    p.add_argument("--start-index", type=int, default=0,
                   help="For logging only; the dataset always appends sequentially.")
    p.add_argument("--repo-id", default=default_repo_id)
    p.add_argument("--root", default=default_root,
                   help="Dataset directory (auto-renamed if it already exists).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no-headless", dest="headless", action="store_false")
    p.add_argument("--image-writer-threads", type=int, default=8)
    p.add_argument("--image-writer-processes", type=int, default=2)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    return p.parse_args()
