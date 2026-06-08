# GraspLens Proposal

## One-Liner

GraspLens is an interpretable runtime assurance layer for learned robotic grasping: it takes candidate grasps from a learned scorer, checks formal safety predicates, probes hidden activations for latent risk concepts, and selects a safer grasp without retraining the original model.

## Motivation

Robotic grasping systems increasingly use learned policies, diffusion grasp generators, or VLA-style models to rank actions. These models can look competent on average while relying on shortcuts that fail under distribution shift. In shelf picking, a shortcut can become physical risk: the robot may select a wrong object, clip a shelf wall, approach through clutter, or choose a grasp on a fragile/hazardous item.

The repository does not yet contain robot logs or grasp demonstrations. The current benchmark therefore uses generated shelf scenes with exact geometry labels. This makes the failure mode, learned representation, and monitoring layer reproducible before connecting the same runtime contract to a ROS2/VLA manipulation stack.

## Experimental Envelope

The benchmark is expressed in metric tabletop coordinates. Normalized scene coordinates map to a `0.72 m x 0.72 m` shelf-picking ROI. The reference hardware envelope is a Franka Research 3-class 7-DoF arm, a Robotiq 2F-85-class parallel-jaw gripper, and an overhead RGB-D camera producing a calibrated top-down occupancy map.

Concrete parameters:

- robot envelope: `855 mm` reach, `3 kg` payload;
- gripper envelope: `85 mm` opening;
- shelf interior: `0.662 m x 0.662 m`;
- object boxes: `54-94 mm` by `40-83 mm`;
- obstacle boxes: `29-58 mm` by `29-72 mm`;
- grasp rectangle: `115.2 mm x 39.6 mm`;
- approach sweep: `18.0-129.6 mm`, seven samples;
- low-clearance threshold: `8.6 mm`;
- default sampling: `10` grasp candidates per object.

These parameters define the measurement envelope for candidate-level predicates. The repository does not contain hardware execution logs or Isaac Sim rollout traces.

## Threat Model

The robot receives a scene observation and a set of candidate grasps. A learned scorer assigns each candidate a utility score. Under distribution shift, the scorer may rank an unsafe grasp highly because it learned a spurious feature.

Failure classes:

- `collision_risk`: gripper intersects shelf bounds or obstacles;
- `wrong_object`: grasp contacts a distractor instead of the requested target;
- `low_clearance`: approach path passes too close to clutter or shelf walls;
- `semantic_hazard`: grasp contacts fragile or hazardous objects;
- `shortcut_failure`: the learned scorer prefers a spurious visual feature over the true target.

## Formal Safety Set

For a scene `x` and candidate grasp set `G(x) = {g_1, ..., g_k}`, define hard predicates:

```text
h_collision(x, g)  >= 0
h_approach(x, g)   >= 0
h_wrong_object(x,g) = 0
h_hazard(x, g)      = 0
```

The conservative geometric safe set is:

```text
S_geo(x) = { g in G(x) : all hard safety predicates pass }
```

The runtime selector chooses:

```text
g*(x) = argmax_g score_theta(x, g)
        subject to g in S_geo(x)
        and unsafe not in C_alpha(z_theta(x, g))
```

`C_alpha` is a split-conformal prediction set built from probe probabilities on hidden activations. The conformal layer is used as calibration for residual learned risk; it does not replace the hard geometric checks.

## Interpretable Monitoring

The learned scorer exposes a hidden activation vector `z_theta(x, g)`. GraspLens trains sparse logistic probes on these activations for:

- collision risk;
- wrong-object risk;
- low-clearance risk;
- semantic hazard;
- aggregate unsafe risk.

The sparse weights provide a compact concept direction. The benchmark also performs activation steering:

```text
z' = z + alpha * v_unsafe
```

and measures whether unsafe candidates move upward in the learned ranking. This gives causal evidence that the learned scorer represents a risk-relevant direction, rather than only a post-hoc label.

## Experiment

The benchmark generates top-down shelf scenes with rectangular products, obstacles, and candidate grasp rectangles. The train distribution makes red objects strongly correlated with the target. The shifted test distribution makes red distractors common.

The learned scorer is trained on a noisy imitation target that overweights the red-object shortcut. On the shifted test set, this produces high-scoring wrong-object and unsafe grasps.

Evaluation compares:

1. `baseline`: max learned score;
2. `geometry`: max score among hard-safe candidates;
3. `probe`: max score among candidates accepted by calibrated probe risk;
4. `grasplens_full`: geometry plus calibrated probe risk.

Metrics:

- unsafe grasp rate;
- wrong-object rate;
- collision rate;
- low-clearance rate;
- semantic-hazard rate;
- success rate;
- utility regret;
- probe AUROC;
- probe sparsity;
- conformal coverage;
- activation patching effect.

## Result

The checked-in run produces a reproducible shortcut failure in learned grasp ranking and compares which runtime checks reject that failure. The result is scoped to generated scenes, candidate-level predicates, learned scoring, sparse probes, conformal calibration, and the final selection rule.

## Path to Isaac and Real Robotics

The generated grasp candidates can later be replaced by:

- candidates from a diffusion grasp generator;
- candidates from a ROS2 `/predict_grasp_pose` service;
- VLA policy action proposals in a simulator such as ManiSkill, LIBERO, or RoboMimic;
- Isaac Sim rollout traces with the same candidate and predicate contract;
- logs from a real shelf-picking robot.

The reusable parts are the runtime contract: candidate grasps in, hard predicates and latent risk estimates in the middle, selected grasp plus explanation out.
