# GraspLens

Interpretable runtime assurance for learned robotic grasping.

GraspLens is a small research demo for the case where there is no real robot dataset yet. It builds a synthetic shelf-picking benchmark, trains a tiny learned grasp scorer that develops a shortcut under distribution shift, and evaluates whether a runtime safety/interp layer can reject unsafe grasps before execution.

The project is designed as a compact model organism for robotics safety:

- a learned grasp scorer can choose high-score but unsafe grasps;
- hard geometric predicates define a conservative safe set;
- sparse activation-level probes recover latent safety concepts;
- split conformal calibration turns probe risk into a runtime rejection rule;
- the final filter chooses a safer grasp without retraining the learned scorer.

## Task

The benchmark represents a top-down shelf-picking task. Each scene contains:

- one target product;
- several distractor products;
- shelf walls and obstacles;
- fragile or hazardous object tags;
- candidate grasp rectangles with approach directions.

The train distribution makes red objects strongly correlated with the target. The shifted test distribution breaks that shortcut by making red distractors common. This creates wrong-object, collision, low-clearance, and semantic-hazard failures.

## Methods

The benchmark compares four selection rules:

| Method | Description |
| --- | --- |
| `baseline` | choose the grasp with the maximum learned score |
| `geometry` | choose the best score among grasps satisfying hard safety predicates |
| `probe` | reject grasps with calibrated latent unsafe risk |
| `grasplens_full` | combine hard geometry and calibrated probe risk |

## Models

`TinyGraspScorer` is a small MLP-style learned scorer. If PyTorch is installed, it uses a PyTorch MLP. In minimal environments without `torch`, it falls back to the included NumPy MLP with the same API and hidden-activation interface.

Sparse concept probes are L1 logistic regressions trained on hidden activations for:

- `collision_risk`
- `wrong_object`
- `low_clearance`
- `semantic_hazard`
- `unsafe`

The probe directions are also used for activation steering/patching experiments.

## Repository Layout

```text
grasplens/
  grasplens/
    scene.py          # synthetic shelf/object generation
    grasps.py         # candidate grasp sampler
    geometry.py       # collision/contact/clearance predicates
    dataset.py        # feature and label construction
    policy.py         # tiny learned scorer
    probes.py         # sparse probes and concept directions
    conformal.py      # split conformal binary prediction sets
    filter.py         # baseline/geometry/probe/full runtime selectors
    metrics.py        # evaluation and activation patching metrics
    visualization.py  # benchmark figures
  experiments/
    run_benchmark.py
    make_figures.py
  outputs/
    results.json
    metrics.csv
    figures/
```

## Setup

```bash
python3 -m pip install -e .
```

Optional PyTorch backend:

```bash
python3 -m pip install -e ".[torch]"
```

## Run

Default benchmark:

```bash
python experiments/run_benchmark.py --train-scenes 2000 --test-scenes 1000 --seed 0
```

Quick smoke test:

```bash
python experiments/run_benchmark.py --train-scenes 100 --test-scenes 50 --epochs 4 --seed 0
```

Outputs are written to:

```text
outputs/results.json
outputs/metrics.csv
outputs/figures/
```

## Default Run Results

The checked-in results were generated with:

```bash
python experiments/run_benchmark.py --train-scenes 2000 --test-scenes 1000 --seed 0
```

| Method | Unsafe rate | Wrong-object rate | Success rate |
| --- | ---: | ---: | ---: |
| `baseline` | 0.918 | 0.861 | 0.078 |
| `geometry` | 0.004 | 0.001 | 0.685 |
| `probe` | 0.642 | 0.315 | 0.349 |
| `grasplens_full` | 0.004 | 0.001 | 0.901 |

Probe AUROC on shifted test candidates is high for the safety concepts: `collision_risk` 0.911, `wrong_object` 0.956, `low_clearance` 0.914, `semantic_hazard` 0.958. Split-conformal test coverage for aggregate unsafe risk is 0.913.

## Current Interpretation

This is not a claim that the system is real-world safe. The useful result is narrower: GraspLens creates a reproducible failure mode where a learned grasp scorer exploits a shortcut, then tests which parts of a runtime assurance layer catch that failure.

The natural next step is to replace synthetic candidates with candidates from a real grasp generator or ROS2 manipulation stack, while keeping the same safety predicates, probe interface, and runtime selection contract.
