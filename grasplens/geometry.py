from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grasplens.grasps import GraspCandidate
from grasplens.scene import Obstacle, ShelfObject, ShelfScene


@dataclass(frozen=True)
class GraspPredicates:
    target_contact: bool
    wrong_object: bool
    collision: bool
    low_clearance: bool
    semantic_hazard: bool
    success: bool
    min_clearance: float
    target_overlap: float
    non_target_overlap: float

    @property
    def unsafe(self) -> bool:
        return self.collision or self.wrong_object or self.low_clearance or self.semantic_hazard

    @property
    def hard_safe(self) -> bool:
        return not self.unsafe


def grasp_polygon(grasp: GraspCandidate, offset: float = 0.0) -> np.ndarray:
    ux = np.array([np.cos(grasp.theta), np.sin(grasp.theta)], dtype=np.float64)
    uy = np.array([-np.sin(grasp.theta), np.cos(grasp.theta)], dtype=np.float64)
    center = np.array([grasp.x, grasp.y], dtype=np.float64) + offset * np.array(grasp.approach)
    local = np.array(
        [
            [-grasp.length / 2, -grasp.width / 2],
            [grasp.length / 2, -grasp.width / 2],
            [grasp.length / 2, grasp.width / 2],
            [-grasp.length / 2, grasp.width / 2],
        ],
        dtype=np.float64,
    )
    return center + local[:, :1] * ux + local[:, 1:] * uy


def _sample_gripper_points(grasp: GraspCandidate, offset: float = 0.0) -> np.ndarray:
    xs = np.linspace(-grasp.length / 2, grasp.length / 2, 9)
    ys = np.linspace(-grasp.width / 2, grasp.width / 2, 5)
    grid = np.array([(x, y) for x in xs for y in ys], dtype=np.float64)
    ux = np.array([np.cos(grasp.theta), np.sin(grasp.theta)], dtype=np.float64)
    uy = np.array([-np.sin(grasp.theta), np.cos(grasp.theta)], dtype=np.float64)
    center = np.array([grasp.x, grasp.y], dtype=np.float64) + offset * np.array(grasp.approach)
    return center + grid[:, :1] * ux + grid[:, 1:] * uy


def _inside_bounds(points: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    return (points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)


def _overlap_fraction(points: np.ndarray, item: ShelfObject | Obstacle) -> float:
    return float(np.mean(_inside_bounds(points, item.bounds)))


def _distance_to_bounds(point: np.ndarray, bounds: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bounds
    dx = max(x0 - point[0], 0.0, point[0] - x1)
    dy = max(y0 - point[1], 0.0, point[1] - y1)
    if dx == 0.0 and dy == 0.0:
        return -min(point[0] - x0, x1 - point[0], point[1] - y0, y1 - point[1])
    return float(np.hypot(dx, dy))


def _min_distance_to_rects(point: np.ndarray, rects: list[tuple[float, float, float, float]]) -> float:
    if not rects:
        return 1.0
    return min(_distance_to_bounds(point, rect) for rect in rects)


def _min_distance_points_to_rects(
    points: np.ndarray,
    rects: list[tuple[float, float, float, float]],
) -> float:
    if not rects:
        return 1.0
    rect_array = np.asarray(rects, dtype=np.float64)
    px = points[:, None, 0]
    py = points[:, None, 1]
    x0 = rect_array[None, :, 0]
    y0 = rect_array[None, :, 1]
    x1 = rect_array[None, :, 2]
    y1 = rect_array[None, :, 3]

    dx = np.maximum(np.maximum(x0 - px, 0.0), px - x1)
    dy = np.maximum(np.maximum(y0 - py, 0.0), py - y1)
    outside_dist = np.hypot(dx, dy)
    inside = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
    inside_margin = -np.minimum.reduce([px - x0, x1 - px, py - y0, y1 - py])
    signed_dist = np.where(inside, inside_margin, outside_dist)
    return float(np.min(signed_dist))


def evaluate_grasp(scene: ShelfScene, grasp: GraspCandidate) -> GraspPredicates:
    points = _sample_gripper_points(grasp)
    inside_shelf = _inside_bounds(points, scene.shelf_bounds)
    shelf_collision = bool(not np.all(inside_shelf))

    target_overlap = 0.0
    non_target_overlap = 0.0
    hazard_overlap = 0.0
    for obj in scene.objects:
        frac = _overlap_fraction(points, obj)
        if obj.is_target:
            target_overlap = max(target_overlap, frac)
        else:
            non_target_overlap = max(non_target_overlap, frac)
        if obj.fragile or obj.hazard:
            hazard_overlap = max(hazard_overlap, frac)

    obstacle_overlap = max((_overlap_fraction(points, obs) for obs in scene.obstacles), default=0.0)
    target_contact = target_overlap > 0.08
    wrong_object = non_target_overlap > 0.10 and non_target_overlap >= target_overlap * 0.65
    semantic_hazard = hazard_overlap > 0.08
    collision = shelf_collision or obstacle_overlap > 0.08

    rects = [obs.bounds for obs in scene.obstacles]
    rects.extend(obj.bounds for obj in scene.objects if not obj.is_target)
    x0, y0, x1, y1 = scene.shelf_bounds
    approach_samples = []
    for offset in np.linspace(0.025, 0.18, 7):
        approach_samples.append(_sample_gripper_points(grasp, offset=offset))
    approach_points = np.concatenate(approach_samples, axis=0)

    shelf_margin = np.min(
        np.stack(
            [
                approach_points[:, 0] - x0,
                x1 - approach_points[:, 0],
                approach_points[:, 1] - y0,
                y1 - approach_points[:, 1],
            ],
            axis=1,
        )
    )
    obstacle_margin = _min_distance_points_to_rects(approach_points, rects)
    min_clearance = float(min(shelf_margin, obstacle_margin))
    low_clearance = min_clearance < 0.012

    unsafe = collision or wrong_object or low_clearance or semantic_hazard
    success = target_contact and not unsafe
    return GraspPredicates(
        target_contact=target_contact,
        wrong_object=wrong_object,
        collision=collision,
        low_clearance=low_clearance,
        semantic_hazard=semantic_hazard,
        success=success,
        min_clearance=min_clearance,
        target_overlap=target_overlap,
        non_target_overlap=non_target_overlap,
    )
