from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grasplens.scene import COLOR_TO_ID, COLORS, ShelfObject, ShelfScene
from grasplens.specs import GRASP_GEOMETRY


@dataclass(frozen=True)
class GraspCandidate:
    candidate_id: int
    scene_id: int
    source_object_id: int
    x: float
    y: float
    theta: float
    length: float = GRASP_GEOMETRY.rectangle_length_norm
    width: float = GRASP_GEOMETRY.rectangle_width_norm

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
        x = obj.x + rng.normal(
            0.0,
            obj.w * GRASP_GEOMETRY.side_jitter_sigma_fraction,
        )
        y = obj.y + obj.h * rng.uniform(*GRASP_GEOMETRY.side_offset_fraction)
        theta = np.pi / 2 + rng.normal(0.0, GRASP_GEOMETRY.angle_jitter_sigma_rad)
    elif side == 1:
        x = obj.x + rng.normal(
            0.0,
            obj.w * GRASP_GEOMETRY.side_jitter_sigma_fraction,
        )
        y = obj.y - obj.h * rng.uniform(*GRASP_GEOMETRY.side_offset_fraction)
        theta = -np.pi / 2 + rng.normal(0.0, GRASP_GEOMETRY.angle_jitter_sigma_rad)
    elif side == 2:
        x = obj.x + obj.w * rng.uniform(*GRASP_GEOMETRY.side_offset_fraction)
        y = obj.y + rng.normal(0.0, obj.h * GRASP_GEOMETRY.side_jitter_sigma_fraction)
        theta = rng.normal(0.0, GRASP_GEOMETRY.angle_jitter_sigma_rad)
    else:
        x = obj.x - obj.w * rng.uniform(*GRASP_GEOMETRY.side_offset_fraction)
        y = obj.y + rng.normal(0.0, obj.h * GRASP_GEOMETRY.side_jitter_sigma_fraction)
        theta = np.pi + rng.normal(0.0, GRASP_GEOMETRY.angle_jitter_sigma_rad)

    if rng.random() < GRASP_GEOMETRY.off_object_jitter_probability:
        x += rng.normal(0.0, GRASP_GEOMETRY.off_object_jitter_sigma_norm)
        y += rng.normal(0.0, GRASP_GEOMETRY.off_object_jitter_sigma_norm)
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
    candidates_per_object: int = GRASP_GEOMETRY.candidates_per_object_default,
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
