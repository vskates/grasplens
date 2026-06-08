from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


COLORS = ("red", "blue", "green", "yellow", "purple")
COLOR_TO_ID = {name: idx for idx, name in enumerate(COLORS)}


@dataclass(frozen=True)
class ShelfObject:
    object_id: int
    label: str
    x: float
    y: float
    w: float
    h: float
    color: str
    fragile: bool = False
    hazard: bool = False

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.x - self.w / 2.0,
            self.y - self.h / 2.0,
            self.x + self.w / 2.0,
            self.y + self.h / 2.0,
        )

    @property
    def is_target(self) -> bool:
        return self.label == "target"


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    w: float
    h: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.x - self.w / 2.0,
            self.y - self.h / 2.0,
            self.x + self.w / 2.0,
            self.y + self.h / 2.0,
        )


@dataclass(frozen=True)
class ShelfScene:
    scene_id: int
    split: str
    objects: tuple[ShelfObject, ...]
    obstacles: tuple[Obstacle, ...]
    shelf_bounds: tuple[float, float, float, float] = (0.04, 0.04, 0.96, 0.96)

    @property
    def target(self) -> ShelfObject:
        for obj in self.objects:
            if obj.is_target:
                return obj
        raise ValueError("scene has no target object")


def _rects_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    pad: float = 0.0,
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (
        ax1 + pad < bx0
        or bx1 + pad < ax0
        or ay1 + pad < by0
        or by1 + pad < ay0
    )


def _sample_object(
    rng: np.random.Generator,
    object_id: int,
    label: str,
    color: str,
    fragile: bool,
    hazard: bool,
    existing: Iterable[ShelfObject],
) -> ShelfObject:
    w = float(rng.uniform(0.075, 0.13))
    h = float(rng.uniform(0.055, 0.115))
    for _ in range(200):
        x = float(rng.uniform(0.16, 0.84))
        y = float(rng.uniform(0.16, 0.84))
        obj = ShelfObject(object_id, label, x, y, w, h, color, fragile, hazard)
        if all(not _rects_overlap(obj.bounds, other.bounds, pad=0.05) for other in existing):
            return obj
    return ShelfObject(object_id, label, x, y, w, h, color, fragile, hazard)


def _target_color(rng: np.random.Generator, split: str) -> str:
    if split == "train":
        return "red" if rng.random() < 0.9 else str(rng.choice(COLORS[1:]))
    return "red" if rng.random() < 0.12 else str(rng.choice(COLORS[1:]))


def _distractor_color(rng: np.random.Generator, split: str, force_red: bool) -> str:
    if force_red:
        return "red"
    if split == "train":
        return "red" if rng.random() < 0.08 else str(rng.choice(COLORS[1:]))
    return "red" if rng.random() < 0.55 else str(rng.choice(COLORS[1:]))


def generate_scene(scene_id: int, split: str, rng: np.random.Generator) -> ShelfScene:
    """Generate a top-down shelf scene with an intentional color shortcut shift.

    In train scenes, the target is usually red. In shifted test scenes, red is
    more often a distractor. This creates a controlled shortcut shift for grasp
    selection.
    """

    if split not in {"train", "test", "calib"}:
        raise ValueError(f"unknown split: {split}")

    effective_split = "train" if split == "calib" else split
    objects: list[ShelfObject] = []
    target = _sample_object(
        rng=rng,
        object_id=0,
        label="target",
        color=_target_color(rng, effective_split),
        fragile=False,
        hazard=False,
        existing=objects,
    )
    objects.append(target)

    n_distractors = int(rng.integers(3, 6))
    force_red_idx = int(rng.integers(0, n_distractors)) if effective_split == "test" else -1
    for idx in range(n_distractors):
        fragile = bool(rng.random() < 0.22)
        hazard = bool(rng.random() < 0.14)
        objects.append(
            _sample_object(
                rng=rng,
                object_id=idx + 1,
                label="distractor",
                color=_distractor_color(rng, effective_split, force_red_idx == idx),
                fragile=fragile,
                hazard=hazard,
                existing=objects,
            )
        )

    obstacles: list[Obstacle] = []
    for _ in range(int(rng.integers(1, 4))):
        for _attempt in range(100):
            obs = Obstacle(
                x=float(rng.uniform(0.13, 0.87)),
                y=float(rng.uniform(0.13, 0.87)),
                w=float(rng.uniform(0.04, 0.08)),
                h=float(rng.uniform(0.04, 0.10)),
            )
            if all(not _rects_overlap(obs.bounds, obj.bounds, pad=0.045) for obj in objects):
                obstacles.append(obs)
                break

    return ShelfScene(
        scene_id=scene_id,
        split=split,
        objects=tuple(objects),
        obstacles=tuple(obstacles),
    )


def generate_scenes(n_scenes: int, split: str, seed: int) -> list[ShelfScene]:
    rng = np.random.default_rng(seed)
    return [generate_scene(i, split, rng) for i in range(n_scenes)]
