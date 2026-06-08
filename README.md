# GraspLens

Interpretable runtime assurance for learned robotic grasping.

GraspLens is an R&D benchmark for learned robotic grasping when the repository does not yet contain robot logs or grasp demonstrations. It generates shelf-picking scenes with exact geometry labels, trains an MLP-style grasp scorer under a controlled distribution shift, and evaluates a runtime assurance layer that rejects unsafe candidates before selection.

The project studies a concrete grasp-selection pipeline:

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

`GraspScorer` is an MLP-style learned scorer. If PyTorch is installed, it uses a PyTorch MLP. In environments without `torch`, it uses the included NumPy implementation with the same API and hidden-activation interface.

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
    scene.py          # generated shelf/object scenes
    grasps.py         # candidate grasp sampler
    geometry.py       # collision/contact/clearance predicates
    dataset.py        # feature and label construction
    policy.py         # MLP-style learned scorer
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

## Current Scope

The checked-in run covers generated scenes, candidate-level geometry predicates, learned scoring, sparse probes, conformal calibration, and runtime selection. Hardware execution, camera noise, controller error, and hazards outside the predicate set are outside the current run.

The natural next step is to replace generated candidates with candidates from a real grasp generator or ROS2 manipulation stack, while keeping the same safety predicates, probe interface, and runtime selection contract.
