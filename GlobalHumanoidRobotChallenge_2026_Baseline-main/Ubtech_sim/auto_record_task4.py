"""Task 4 - Packing_Box auto-recorder.

Picks the foam at world (0.856, 0.185, 1.04) and packs it into the box at
(0.855, 0.30, 1.13). Single-arm pick-and-place with extended gripper
(tcp_offset z = 0.22). Combine with the 50 baseline episodes shipped by the
organizers to reach >= 1500 total.

Run:
    /isaac-sim/python.sh -m Ubtech_sim.auto_record_task4 \
        --num-episodes 1500 --root /workspace/datasets/task4_packing_box
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


TASK_NAME = "Pick the foam piece and pack it into the box."
TASK_YAML = os.path.join(HERE, "config", "Packing_Box.yaml")
DEFAULT_REPO_ID = "ghrc2026/task4_packing_box"
DEFAULT_ROOT = os.path.join(HERE, "..", "datasets", "task4_packing_box")

BOX_WORLD = np.array([0.855, 0.30, 1.13])
NUM_PARTS_PER_EPISODE = 1


def make_place_target_fn(runtime: SimRuntime):
    def fn(part_idx: int, side: str, grasp_pose_base: np.ndarray) -> np.ndarray:
        world = BOX_WORLD.copy()
        world[2] += 0.03
        base = runtime.coord_transform.world_to_robot(world)
        target = np.zeros(6, dtype=float)
        target[:3] = base - runtime.planner.tcp_offset_base
        target[3:] = grasp_pose_base[3:].copy()
        return target
    return fn


def randomize_seed(runtime: SimRuntime, ep_idx: int) -> None:
    np.random.seed(ep_idx + 3001)


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
        randomize_fn=randomize_seed,
        timings=FSMTimings(
            approach=35, descend=28, grasp=22, lift=28,
            transport=45, place=28, release=18, retreat=28,
        ),
        max_extra_frames=90,
    )

    try:
        for i in range(args.num_episodes):
            ok = run_episode(runtime, recorder, spec, ep_idx=args.start_index + i)
            if not ok:
                print(f"[task4] ep {i} failed, continuing...")
            if (i + 1) % 25 == 0:
                print(f"[task4] {i + 1}/{args.num_episodes} episodes saved")
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    main()
