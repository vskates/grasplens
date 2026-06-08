from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ReferenceHardwareSpec:
    arm_class: str = "Franka Research 3-class 7-DoF arm"
    arm_dof: int = 7
    arm_reach_m: float = 0.855
    arm_payload_kg: float = 3.0
    gripper_class: str = "Robotiq 2F-85-class parallel-jaw gripper"
    gripper_opening_m: float = 0.085
    camera_class: str = "overhead RGB-D camera"
    camera_stream: str = "calibrated top-down occupancy map"


@dataclass(frozen=True)
class SceneScaleSpec:
    workspace_size_m: float = 0.72
    shelf_bounds_norm: tuple[float, float, float, float] = (0.04, 0.04, 0.96, 0.96)
    object_width_norm: tuple[float, float] = (0.075, 0.13)
    object_height_norm: tuple[float, float] = (0.055, 0.115)
    object_center_norm: tuple[float, float] = (0.16, 0.84)
    object_separation_pad_norm: float = 0.05
    obstacle_width_norm: tuple[float, float] = (0.04, 0.08)
    obstacle_height_norm: tuple[float, float] = (0.04, 0.10)
    obstacle_center_norm: tuple[float, float] = (0.13, 0.87)
    obstacle_separation_pad_norm: float = 0.045


@dataclass(frozen=True)
class GraspGeometrySpec:
    rectangle_length_norm: float = 0.16
    rectangle_width_norm: float = 0.055
    side_jitter_sigma_fraction: float = 0.16
    side_offset_fraction: tuple[float, float] = (0.0, 0.36)
    angle_jitter_sigma_rad: float = 0.28
    off_object_jitter_probability: float = 0.18
    off_object_jitter_sigma_norm: float = 0.085
    candidates_per_object_default: int = 10


@dataclass(frozen=True)
class PredicateThresholdSpec:
    target_contact_overlap_fraction: float = 0.08
    wrong_object_overlap_fraction: float = 0.10
    wrong_object_relative_overlap: float = 0.65
    hazard_overlap_fraction: float = 0.08
    collision_overlap_fraction: float = 0.08
    approach_sweep_offsets_norm: tuple[float, float, int] = (0.025, 0.18, 7)
    low_clearance_norm: float = 0.012


@dataclass(frozen=True)
class ValidationScopeSpec:
    included_artifact: str = "offline generated-scene benchmark with exact candidate labels"
    isaac_ros2_status: str = "interface contract specified; no Isaac or robot rollout logs are included"
    measured_unit: str = "per-scene selected grasp predicate outcome"


REFERENCE_HARDWARE = ReferenceHardwareSpec()
SCENE_SCALE = SceneScaleSpec()
GRASP_GEOMETRY = GraspGeometrySpec()
PREDICATE_THRESHOLDS = PredicateThresholdSpec()
VALIDATION_SCOPE = ValidationScopeSpec()


def norm_to_m(value: float) -> float:
    return float(value) * SCENE_SCALE.workspace_size_m


def norm_interval_to_m(values: tuple[float, float]) -> tuple[float, float]:
    return (norm_to_m(values[0]), norm_to_m(values[1]))


def _rounded_m(value: float) -> float:
    return round(float(value), 4)


def benchmark_spec_payload(candidates_per_object: int | None = None) -> dict[str, Any]:
    x0, y0, x1, y1 = SCENE_SCALE.shelf_bounds_norm
    approach_start, approach_stop, approach_steps = PREDICATE_THRESHOLDS.approach_sweep_offsets_norm
    payload: dict[str, Any] = {
        "reference_hardware": asdict(REFERENCE_HARDWARE),
        "scene_scale": {
            **asdict(SCENE_SCALE),
            "shelf_interior_m": [
                _rounded_m((x1 - x0) * SCENE_SCALE.workspace_size_m),
                _rounded_m((y1 - y0) * SCENE_SCALE.workspace_size_m),
            ],
            "object_width_m": list(map(_rounded_m, norm_interval_to_m(SCENE_SCALE.object_width_norm))),
            "object_height_m": list(map(_rounded_m, norm_interval_to_m(SCENE_SCALE.object_height_norm))),
            "obstacle_width_m": list(map(_rounded_m, norm_interval_to_m(SCENE_SCALE.obstacle_width_norm))),
            "obstacle_height_m": list(map(_rounded_m, norm_interval_to_m(SCENE_SCALE.obstacle_height_norm))),
        },
        "grasp_geometry": {
            **asdict(GRASP_GEOMETRY),
            "rectangle_length_m": _rounded_m(norm_to_m(GRASP_GEOMETRY.rectangle_length_norm)),
            "rectangle_width_m": _rounded_m(norm_to_m(GRASP_GEOMETRY.rectangle_width_norm)),
            "off_object_jitter_sigma_m": _rounded_m(norm_to_m(GRASP_GEOMETRY.off_object_jitter_sigma_norm)),
        },
        "predicate_thresholds": {
            **asdict(PREDICATE_THRESHOLDS),
            "approach_sweep_offsets_m": [
                _rounded_m(norm_to_m(approach_start)),
                _rounded_m(norm_to_m(approach_stop)),
                approach_steps,
            ],
            "low_clearance_m": _rounded_m(norm_to_m(PREDICATE_THRESHOLDS.low_clearance_norm)),
        },
        "validation_scope": asdict(VALIDATION_SCOPE),
    }
    if candidates_per_object is not None:
        payload["grasp_geometry"]["candidates_per_object"] = int(candidates_per_object)
    return payload
