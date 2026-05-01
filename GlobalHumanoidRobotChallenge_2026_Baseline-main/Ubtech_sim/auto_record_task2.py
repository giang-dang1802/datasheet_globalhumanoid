"""Task 2 - Conveyor_Sorting auto-recorder.

Parts move along the conveyor (+X). They are sorted by class:
    Task2_Part_A  -> right bin (x ~= 0.886)
    other (Part_B) -> left bin  (x ~= 0.486)
GraspPlanner already supports `belt_velocity` + `lookahead_s` for predictive
tracking, configured via Conveyor_Sorting.yaml.

Run:
    /isaac-sim/python.sh -m Ubtech_sim.auto_record_task2 \
        --num-episodes 1500 --root /workspace/datasets/task2_conveyor_sorting
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from auto_record_common import (  # noqa: E402
    EpisodeSpec,
    FSMTimings,
    LeRobotRecorder,
    SimRuntime,
    SimRuntimeConfig,
    parse_cli,
    run_episode,
)


TASK_NAME = "Sort moving parts from conveyor into left or right bin by class."
TASK_YAML = os.path.join(HERE, "config", "Conveyor_Sorting.yaml")
DEFAULT_REPO_ID = "ghrc2026/task2_conveyor_sorting"
DEFAULT_ROOT = os.path.join(HERE, "..", "datasets", "task2_conveyor_sorting")

BOX_LEFT_WORLD = np.array([0.486, -0.062, 0.95])
BOX_RIGHT_WORLD = np.array([0.886, -0.062, 0.95])
NUM_PARTS_PER_EPISODE = 4


def _classify_part(prim_path: str) -> str:
    return "A" if "task2_part_a" in (prim_path or "").lower() else "B"


def make_place_target_fn(runtime: SimRuntime):
    def fn(part_idx: int, side: str, grasp_pose_base: np.ndarray) -> np.ndarray:
        prim = runtime.planner.target_prim_path or ""
        cls = _classify_part(prim)
        world = BOX_RIGHT_WORLD.copy() if cls == "A" else BOX_LEFT_WORLD.copy()
        # Fan out within the box so consecutive drops do not stack.
        world[1] += (part_idx - NUM_PARTS_PER_EPISODE / 2.0) * 0.05
        world[2] += 0.06
        base = runtime.coord_transform.world_to_robot(world)
        target = np.zeros(6, dtype=float)
        target[:3] = base - runtime.planner.tcp_offset_base
        target[3:] = grasp_pose_base[3:].copy()
        return target
    return fn


def randomize_belt(runtime: SimRuntime, ep_idx: int) -> None:
    """Per-episode jitter on belt speed (planner reads this each step)."""
    base_v = 0.10
    jitter = float(np.random.uniform(-0.02, 0.02))
    runtime.task_cfg.setdefault("grasp", {})
    runtime.task_cfg["grasp"]["belt_velocity"] = [base_v + jitter, 0.0, 0.0]
    runtime.planner.grasp_cfg["belt_velocity"] = [base_v + jitter, 0.0, 0.0]


def main() -> None:
    args = parse_cli(DEFAULT_REPO_ID, DEFAULT_ROOT)

    runtime = SimRuntime(SimRuntimeConfig(
        task_yaml=TASK_YAML,
        headless=args.headless,
        fps=args.fps,
        seed=args.seed,
    ))
    runtime.boot()

    recorder = LeRobotRecorder(
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    spec = EpisodeSpec(
        task_name=TASK_NAME,
        num_parts=NUM_PARTS_PER_EPISODE,
        place_target_fn=make_place_target_fn(runtime),
        randomize_fn=randomize_belt,
        timings=FSMTimings(
            approach=24, descend=18, grasp=14, lift=18,
            transport=32, place=20, release=12, retreat=18,
        ),
    )

    try:
        for i in range(args.num_episodes):
            ok = run_episode(runtime, recorder, spec, ep_idx=args.start_index + i)
            if not ok:
                print(f"[task2] ep {i} failed, continuing...")
            if (i + 1) % 25 == 0:
                print(f"[task2] {i + 1}/{args.num_episodes} episodes saved")
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    main()
