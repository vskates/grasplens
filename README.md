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

## Experimental Envelope

The benchmark is specified in metric tabletop coordinates rather than pixel-only image space. Normalized scene coordinates map to a `0.72 m x 0.72 m` top-down shelf ROI.

| Component | Reference used in this benchmark |
| --- | --- |
| Robot envelope | Franka Research 3-class 7-DoF arm, `855 mm` reach, `3 kg` payload |
| Gripper envelope | Robotiq 2F-85-class parallel-jaw gripper, `85 mm` opening |
| Perception envelope | Overhead RGB-D camera producing a calibrated top-down occupancy map |
| Shelf ROI | `0.72 m x 0.72 m` workspace; `0.662 m x 0.662 m` shelf interior |
| Object boxes | `54-94 mm` width, `40-83 mm` height |
| Obstacle boxes | `29-58 mm` width, `29-72 mm` height |
| Grasp rectangle | `115.2 mm x 39.6 mm` |
| Approach sweep | `18.0-129.6 mm`, seven samples along the approach vector |
| Clearance gate | minimum approach clearance `>= 8.6 mm` |

The reference envelope is used to set scene scale and predicate thresholds. The checked-in metrics are from the offline generated-scene benchmark. Robot logs and Isaac Sim rollouts are not included in this repository.

Public hardware references for the envelope: [Franka Research 3](https://franka.de/products/franka-research-3), [Robotiq 2F-85](https://robotiq.com/products/2f85-140-adaptive-robot-gripper), and [Intel RealSense D435](https://www.intelrealsense.com/depth-camera-d435/).

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
    specs.py          # metric robot/scene/grasp/predicate envelope
    visualization.py  # benchmark figures
  experiments/
    run_benchmark.py
    make_figures.py
  outputs/
    results.json
    benchmark_spec.json
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
outputs/benchmark_spec.json
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

The checked-in run covers generated scenes, metric candidate geometry, candidate-level predicates, learned scoring, sparse probes, conformal calibration, and runtime selection. Hardware execution logs, Isaac Sim rollout traces, camera noise, controller error, and hazards outside the predicate set are outside the current run.

The natural next step is to replace generated candidates with candidates from a real grasp generator or ROS2 manipulation stack, while keeping the same safety predicates, probe interface, and runtime selection contract.
