"""Task 3 - Foam_Inlaying auto-recorder.

6 parts (3 per class) in two side bins are inserted into holes on the foam in
the centre of the table at world (0.76, 0.30, 1.04).
tcp_offset = (0, 0, 0.22) because the gripper is extended.

Run:
    /isaac-sim/python.sh -m Ubtech_sim.auto_record_task3 \
        --num-episodes 1500 --root /workspace/datasets/task3_foam_inlaying
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


TASK_NAME = "Insert each part from side bins into the foam holes on the table."
TASK_YAML = os.path.join(HERE, "config", "Foam_Inlaying.yaml")
DEFAULT_REPO_ID = "ghrc2026/task3_foam_inlaying"
DEFAULT_ROOT = os.path.join(HERE, "..", "datasets", "task3_foam_inlaying")

FOAM_WORLD = np.array([0.76, 0.30, 1.04])
NUM_PARTS_PER_EPISODE = 6


def make_place_target_fn(runtime: SimRuntime):
    """Foam holes assumed to lie on a 3x2 grid centred on FOAM_WORLD."""
    def fn(part_idx: int, side: str, grasp_pose_base: np.ndarray) -> np.ndarray:
        col = part_idx % 3
        row = part_idx // 3 % 2
        offset = np.array([(col - 1) * 0.06, (row - 0.5) * 0.06, 0.0])
        world = FOAM_WORLD + offset
        base = runtime.coord_transform.world_to_robot(world)
        target = np.zeros(6, dtype=float)
        target[:3] = base - runtime.planner.tcp_offset_base
        target[3:] = grasp_pose_base[3:].copy()
        return target
    return fn


def randomize_seed(runtime: SimRuntime, ep_idx: int) -> None:
    np.random.seed(ep_idx + 7919)


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
            approach=30, descend=24, grasp=18, lift=24,
            transport=40, place=30, release=14, retreat=24,
        ),
    )

    try:
        for i in range(args.num_episodes):
            ok = run_episode(runtime, recorder, spec, ep_idx=args.start_index + i)
            if not ok:
                print(f"[task3] ep {i} failed, continuing...")
            if (i + 1) % 25 == 0:
                print(f"[task3] {i + 1}/{args.num_episodes} episodes saved")
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    main()
