"""Task 1 - Part_Sorting auto-recorder.

Picks 4 parts from the table and places them inside the box at world
(1.20, 0.30, 1.05). Outputs a LeRobotDataset V2.1 ready for ACT / Pi0.

Run:
    /isaac-sim/python.sh -m Ubtech_sim.auto_record_task1 \
        --num-episodes 1500 --root /workspace/datasets/task1_part_sorting
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


TASK_NAME = "Pick parts from the table and place them into the sorting box."
TASK_YAML = os.path.join(HERE, "config", "Part_Sorting.yaml")
DEFAULT_REPO_ID = "ghrc2026/task1_part_sorting"
DEFAULT_ROOT = os.path.join(HERE, "..", "datasets", "task1_part_sorting")

BOX_WORLD = np.array([1.20, 0.30, 1.10])
NUM_PARTS_PER_EPISODE = 4


def make_place_target_fn(runtime: SimRuntime):
    def fn(part_idx: int, side: str, grasp_pose_base: np.ndarray) -> np.ndarray:
        gx = part_idx % 2
        gy = part_idx // 2 % 2
        offset = np.array([(gx - 0.5) * 0.10, (gy - 0.5) * 0.10, 0.06])
        world = BOX_WORLD + offset
        base = runtime.coord_transform.world_to_robot(world)
        target = np.zeros(6, dtype=float)
        target[:3] = base - runtime.planner.tcp_offset_base
        target[3:] = grasp_pose_base[3:].copy()
        return target
    return fn


def randomize_lights(runtime: SimRuntime, ep_idx: int) -> None:
    """Light DR: spawn one extra distant light with random params per episode."""
    try:
        import omni.replicator.core as rep
        with rep.trigger.on_frame(num_frames=1):
            rep.create.light(
                light_type="distant",
                intensity=rep.distribution.uniform(800, 1500),
                rotation=rep.distribution.uniform((-30, -30, -30), (30, 30, 30)),
            )
    except Exception as exc:
        print(f"[task1] light randomize skipped: {exc}")


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
        randomize_fn=randomize_lights,
        timings=FSMTimings(
            approach=30, descend=22, grasp=18, lift=22,
            transport=35, place=22, release=14, retreat=22,
        ),
    )

    try:
        for i in range(args.num_episodes):
            ok = run_episode(runtime, recorder, spec, ep_idx=args.start_index + i)
            if not ok:
                print(f"[task1] ep {i} failed, continuing...")
            if (i + 1) % 25 == 0:
                print(f"[task1] {i + 1}/{args.num_episodes} episodes saved")
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    main()
