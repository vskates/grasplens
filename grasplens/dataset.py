from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from grasplens.geometry import GraspPredicates, evaluate_grasp
from grasplens.grasps import GraspCandidate, sample_grasps
from grasplens.scene import COLORS, ShelfObject, ShelfScene


@dataclass
class BenchmarkDataset:
    scenes: list[ShelfScene]
    candidates: list[GraspCandidate]
    predicates: list[GraspPredicates]
    x: np.ndarray
    y: np.ndarray
    scene_index: np.ndarray
    source_object_id: np.ndarray
    labels: dict[str, np.ndarray]
    feature_names: list[str]


def _source_object(scene: ShelfScene, candidate: GraspCandidate) -> ShelfObject:
    for obj in scene.objects:
        if obj.object_id == candidate.source_object_id:
            return obj
    raise KeyError(candidate.source_object_id)


def feature_vector(scene: ShelfScene, candidate: GraspCandidate) -> tuple[np.ndarray, list[str]]:
    obj = _source_object(scene, candidate)
    x0, y0, x1, y1 = scene.shelf_bounds
    dist_left = candidate.x - x0
    dist_right = x1 - candidate.x
    dist_bottom = candidate.y - y0
    dist_top = y1 - candidate.y
    rel_x = candidate.x - obj.x
    rel_y = candidate.y - obj.y
    nearest_obstacle = 1.0
    for obs in scene.obstacles:
        dx = max(abs(candidate.x - obs.x) - obs.w / 2.0, 0.0)
        dy = max(abs(candidate.y - obs.y) - obs.h / 2.0, 0.0)
        nearest_obstacle = min(nearest_obstacle, float(np.hypot(dx, dy)))

    color_features = [float(obj.color == color) for color in COLORS]
    base = [
        candidate.x,
        candidate.y,
        np.cos(candidate.theta),
        np.sin(candidate.theta),
        candidate.length,
        candidate.width,
        rel_x,
        rel_y,
        abs(rel_x),
        abs(rel_y),
        obj.x,
        obj.y,
        obj.w,
        obj.h,
        float(obj.is_target),
        float(obj.fragile),
        float(obj.hazard),
        dist_left,
        dist_right,
        dist_bottom,
        dist_top,
        nearest_obstacle,
        float(len(scene.objects)),
        float(sum(o.color == "red" for o in scene.objects)),
        float(sum(o.fragile or o.hazard for o in scene.objects)),
        scene.target.x,
        scene.target.y,
    ]
    names = [
        "grasp_x",
        "grasp_y",
        "grasp_cos_theta",
        "grasp_sin_theta",
        "grasp_length",
        "grasp_width",
        "rel_obj_x",
        "rel_obj_y",
        "abs_rel_obj_x",
        "abs_rel_obj_y",
        "obj_x",
        "obj_y",
        "obj_w",
        "obj_h",
        "semantic_match_target",
        "obj_fragile",
        "obj_hazard",
        "dist_left_wall",
        "dist_right_wall",
        "dist_bottom_wall",
        "dist_top_wall",
        "nearest_obstacle_dist",
        "num_objects",
        "num_red_objects",
        "num_fragile_or_hazard",
        "target_x",
        "target_y",
    ]
    names.extend([f"obj_color_{color}" for color in COLORS])
    return np.asarray(base + color_features, dtype=np.float32), names


def imitation_target(
    scene: ShelfScene,
    candidate: GraspCandidate,
    predicates: GraspPredicates,
    rng: np.random.Generator,
) -> float:
    """Noisy reward that overweights a spurious red-object shortcut."""

    obj = _source_object(scene, candidate)
    centered = np.exp(-((candidate.x - obj.x) ** 2 + (candidate.y - obj.y) ** 2) / (2 * 0.055**2))
    red_shortcut = float(obj.color == "red")
    score = (
        1.05 * float(obj.is_target)
        + 2.15 * red_shortcut
        + 0.40 * float(predicates.target_contact)
        + 0.35 * centered
        - 0.25 * float(predicates.collision)
        - 0.15 * float(predicates.low_clearance)
        - 0.22 * float(obj.fragile or obj.hazard)
        - 0.12 * float(predicates.wrong_object)
    )
    return float(score + rng.normal(0.0, 0.13))


def build_dataset(
    scenes: list[ShelfScene],
    seed: int,
    candidates_per_object: int = 10,
    progress_desc: str | None = None,
) -> BenchmarkDataset:
    scenes = list(scenes)
    rng = np.random.default_rng(seed)
    candidates: list[GraspCandidate] = []
    predicates: list[GraspPredicates] = []
    features: list[np.ndarray] = []
    targets: list[float] = []
    scene_index: list[int] = []
    source_object_id: list[int] = []
    feature_names: list[str] | None = None

    iterable = tqdm(scenes, desc=progress_desc) if progress_desc else scenes
    for local_scene_idx, scene in enumerate(iterable):
        scene_candidates = sample_grasps(scene, rng, candidates_per_object=candidates_per_object)
        for cand in scene_candidates:
            pred = evaluate_grasp(scene, cand)
            x, names = feature_vector(scene, cand)
            if feature_names is None:
                feature_names = names
            candidates.append(cand)
            predicates.append(pred)
            features.append(x)
            targets.append(imitation_target(scene, cand, pred, rng))
            scene_index.append(local_scene_idx)
            source_object_id.append(cand.source_object_id)

    labels = {
        "collision_risk": np.asarray([p.collision for p in predicates], dtype=np.int64),
        "wrong_object": np.asarray([p.wrong_object for p in predicates], dtype=np.int64),
        "low_clearance": np.asarray([p.low_clearance for p in predicates], dtype=np.int64),
        "semantic_hazard": np.asarray([p.semantic_hazard for p in predicates], dtype=np.int64),
        "success": np.asarray([p.success for p in predicates], dtype=np.int64),
        "unsafe": np.asarray([p.unsafe for p in predicates], dtype=np.int64),
        "target_contact": np.asarray([p.target_contact for p in predicates], dtype=np.int64),
    }
    return BenchmarkDataset(
        scenes=scenes,
        candidates=candidates,
        predicates=predicates,
        x=np.vstack(features).astype(np.float32),
        y=np.asarray(targets, dtype=np.float32),
        scene_index=np.asarray(scene_index, dtype=np.int64),
        source_object_id=np.asarray(source_object_id, dtype=np.int64),
        labels=labels,
        feature_names=feature_names or [],
    )
