from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grasplens.scene import COLOR_TO_ID, COLORS, ShelfObject, ShelfScene


@dataclass(frozen=True)
class GraspCandidate:
    candidate_id: int
    scene_id: int
    source_object_id: int
    x: float
    y: float
    theta: float
    length: float = 0.16
    width: float = 0.055

    @property
    def approach(self) -> tuple[float, float]:
        return (-float(np.cos(self.theta)), -float(np.sin(self.theta)))


def _jittered_grasp(
    rng: np.random.Generator,
    scene_id: int,
    candidate_id: int,
    obj: ShelfObject,
) -> GraspCandidate:
    side = int(rng.integers(0, 4))
    if side == 0:
        x = obj.x + rng.normal(0.0, obj.w * 0.16)
        y = obj.y + obj.h * rng.uniform(0.0, 0.36)
        theta = np.pi / 2 + rng.normal(0.0, 0.28)
    elif side == 1:
        x = obj.x + rng.normal(0.0, obj.w * 0.16)
        y = obj.y - obj.h * rng.uniform(0.0, 0.36)
        theta = -np.pi / 2 + rng.normal(0.0, 0.28)
    elif side == 2:
        x = obj.x + obj.w * rng.uniform(0.0, 0.36)
        y = obj.y + rng.normal(0.0, obj.h * 0.16)
        theta = rng.normal(0.0, 0.28)
    else:
        x = obj.x - obj.w * rng.uniform(0.0, 0.36)
        y = obj.y + rng.normal(0.0, obj.h * 0.16)
        theta = np.pi + rng.normal(0.0, 0.28)

    if rng.random() < 0.18:
        x += rng.normal(0.0, 0.085)
        y += rng.normal(0.0, 0.085)
    return GraspCandidate(
        candidate_id=candidate_id,
        scene_id=scene_id,
        source_object_id=obj.object_id,
        x=float(x),
        y=float(y),
        theta=float(((theta + np.pi) % (2 * np.pi)) - np.pi),
    )


def sample_grasps(
    scene: ShelfScene,
    rng: np.random.Generator,
    candidates_per_object: int = 10,
) -> list[GraspCandidate]:
    candidates: list[GraspCandidate] = []
    next_id = 0
    for obj in scene.objects:
        for _ in range(candidates_per_object):
            candidates.append(_jittered_grasp(rng, scene.scene_id, next_id, obj))
            next_id += 1
    return candidates


def object_feature_vector(obj: ShelfObject) -> np.ndarray:
    color = np.zeros(len(COLORS), dtype=np.float32)
    color[COLOR_TO_ID[obj.color]] = 1.0
    base = np.array(
        [
            obj.x,
            obj.y,
            obj.w,
            obj.h,
            float(obj.is_target),
            float(obj.fragile),
            float(obj.hazard),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, color]).astype(np.float32)
