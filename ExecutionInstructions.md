# Execution Instructions — PDDL Shepherding

## Project Overview

This project simulates a **shepherding robot** that guides a flock of sheep to a goal corral.
It uses a **two-layer hybrid architecture**:

- **Reactive layer** — `ShepherdRobot` + `Sheep` (continuous physics, runs every step)
- **Deliberative layer** — PDDL planner (`pyperplan`) decides *what to do* at a symbolic level

```
Continuous state  →  WorldAbstraction  →  problem.pddl  →  pyperplan  →  plan
      ↑                                                                     ↓
 Sheep / Robot  ←─────────────────  PlanExecutor (mode + param overrides)  ┘
```

---

## Repository Structure

```
shepherding-pyrobosim/
├── config/
│   └── world.yaml               # World bounds, obstacles, goal
│   └── pddl_experiment.yaml     # Main experiment configuration
│   └── fixed_experiment.yaml    # Reproducible fixed-start example profile
├── data/logs/
│   ├── run.json                 # Log from original reactive experiment
│   └── run_pddl.json            # Log from PDDL-guided experiment
├── experiments/
│   ├── run_experiment.py        # Original reactive-only experiment
│   ├── run_pddl_experiment.py   # PDDL-guided experiment  ← main entry point
│   ├── plot_run.py              # Plot static trajectory from a log
│   └── plot_run_video.py        # Export trajectory as a video
├── pddl/
│   ├── shepherding_domain.pddl  # PDDL domain (actions, predicates)
│   ├── problem.pddl             # Auto-generated at runtime
│   └── problem_generator.py    # Builds problem.pddl from abstract state
├── shepherding/
│   ├── sheep.py                 # Sheep flocking model (Reynolds rules)
│   ├── robot.py                 # Shepherd robot reactive controller
│   ├── world_abstraction.py     # Continuous → discrete zone mapping
│   ├── planner_interface.py     # Calls pyperplan, parses plan output
│   └── plan_executor.py        # Maps PDDL actions to robot params/modes
└── sim_logging/
    └── logger.py                # JSON run logger
```

---

## 1. Installation

### Requirements

- Python 3.8 or later
- pip

### Create and activate a virtual environment (venv)

From the project root:

```bash
cd /path/to/shepherding-pyrobosim
python3 -m venv .venv
source .venv/bin/activate
```

If activation succeeds, your shell prompt should show `(.venv)`.

To deactivate later:

```bash
deactivate
```

### Install dependencies

With the virtual environment activated:

```bash
pip install -r requirements.txt
```

> `pyperplan` is the PDDL planner used. It is pure-Python and requires no
> external binaries.

Verify pyperplan is available:

```bash
python3 -m pyperplan --help
```

---

## 2. Running the Experiments

### 2.1 PDDL-guided experiment (recommended)

```bash
cd /path/to/shepherding-pyrobosim
python3 experiments/run_pddl_experiment.py
```

Expected console output:

```
[Planner] Re-planning ...
[Planner] Plan length: 8 actions
  1. move-robot z_0_0 z_1_0
  2. collect-outlier sheep3 z_1_0 z_2_1
  ...
[Sim] Step 312: action 'collect-outlier' completed.
[Sim] Next action: drive-flock z_2_1 z_2_2 z_3_3
[Planner] Re-planning ...
...
[Sim] Goal reached at step 4821!
[Logger] Saved log to data/logs/run_pddl.json
[Sim] Done.
```

### 2.2 Original reactive-only experiment (baseline)

```bash
python3 experiments/run_experiment.py
```

Log saved to `data/logs/run.json`.

### 2.3 Run with an alternative config file

Both experiment scripts accept an optional config path via the
`EXPERIMENT_CONFIG` environment variable.

Use the reproducible example profile:

```bash
EXPERIMENT_CONFIG=config/fixed_experiment.yaml \
python3 experiments/run_experiment.py

EXPERIMENT_CONFIG=config/fixed_experiment.yaml \
python3 experiments/run_pddl_experiment.py

# POPF + all-robots PDDL profile
EXPERIMENT_CONFIG=config/popf_experiment.yaml \
python3 experiments/run_pddl_experiment.py
```

---

## 3. Key Configuration Parameters

All parameters are loaded from `config/pddl_experiment.yaml`.
Both `experiments/run_experiment.py` and `experiments/run_pddl_experiment.py`
read from this same YAML file.

### Simulation

| Parameter | Default | Description |
|---|---|---|
| `NUM_SHEEP` | `15` | Number of sheep in the flock |
| `STEPS` | `null` | Maximum simulation steps (`null` means run until goal is reached) |
| `seed` | `null` or integer | Random seed (`null` = non-deterministic runs, integer = reproducible runs) |
| `robot_start_mode` | `origin` | Robot spawn strategy: `origin`, `random`, or `fixed` |
| `robot_start_positions` | `[[0.0, 0.0]]` | Per-robot start coordinates used when `robot_start_mode: fixed` |
| `sheep_start_mode` | `random` | Sheep spawn strategy: `random` or `fixed` |
| `sheep_start_positions` | `[]` | Per-sheep start coordinates used when `sheep_start_mode: fixed` |
| `sheep_spawn_bounds` | `[0.0,0.0,10.0,10.0]` | Sheep random spawn area `[xmin,ymin,xmax,ymax]` |
| `BOUNDS` | `[0,0,20,20]` | World boundaries (xmin, ymin, xmax, ymax) |
| `GOAL_POS` | `[18.0, 18.0]` | Target corral position |
| `GOAL_RADIUS` | `2.0` | Distance threshold to consider goal reached |
| `coordination_mode` | `roles` | Multi-robot mode: `roles` (alpha+collector+flankers) or `pddl_all_robots` (each robot runs PDDL) |

### PDDL Planner

| Parameter | Default | Description |
|---|---|---|
| `REPLAN_INTERVAL` | `300` | Steps between forced re-planning calls |
| `GRID_SIZE` | `5` | NxN zone grid resolution (5 → 25 zones) |

### Flocking (sheep behaviour)

| Parameter | Default | Description |
|---|---|---|
| `w_coh` | `0.05` | Cohesion weight |
| `w_sep` | `0.4` | Separation weight |
| `w_align` | `0.05` | Alignment weight |
| `w_robot` | `0.2` | Robot avoidance weight |
| `w_fence` | `1.5` | Fence repulsion weight |
| `max_speed` | `0.15` | Maximum sheep speed (units/step) |
| `neighbor_radius` | `2.5` | Radius for local flock perception |
| `noise_std` | `0.01` | Random motion noise standard deviation for sheep |

### Robot behaviour

| Parameter | Default | Description |
|---|---|---|
| `collect_threshold` | `3.0` | Max spread before switching to collect mode |
| `collect_distance` | `1.5` | How far behind an outlier the robot targets |
| `drive_distance` | `2.0` | How far behind the flock the robot positions |
| `robot_max_speed` | `0.3` | Maximum robot speed (units/step) |
| `robot_influence` | `3.0` | Radius within which the robot repels sheep |
| `robot_sheep_min_distance` | `1.0` | Minimum distance robot tries to keep from sheep |
| `robot_fence_clearance` | `0.35` | Minimum robot distance to the fence segment |

### Logging and metrics

| Parameter | Default | Description |
|---|---|---|
| `planner_quiet` | `true` | If `true`, suppresses detailed planner action dumps |
| `iteration_log_interval` | `100` | Prints iteration progress every N steps (`0` disables periodic print) |
| `metrics.enabled` | `true` | Enables run summary metrics JSON output |
| `metrics.reactive_output_path` | `data/logs/run_metrics.json` | Metrics output path for reactive-only run |
| `metrics.pddl_output_path` | `data/logs/run_pddl_metrics.json` | Metrics output path for PDDL run |

---

## 4. Configuration Examples

### Example A — Small, tight flock (fast convergence)

Good for quickly validating that the planner works.

```yaml
simulation:
  num_sheep: 8
  steps: 8000

planner:
  replan_interval: 200
  grid_size: 4          # coarser grid -> shorter plans

robot:
  collect_threshold: 2.0

flocking:
  w_coh: 0.1            # stronger cohesion
  max_speed: 0.1
```

### Example B — Large dispersed flock (stress test)

Tests the `collect-outlier` action path heavily.

```yaml
simulation:
  num_sheep: 30
  steps: 20000

planner:
  replan_interval: 500
  grid_size: 6

robot:
  collect_threshold: 4.0

flocking:
  w_sep: 0.5             # sheep spread out more
  neighbor_radius: 1.5
```

### Example C — Fine-grained planning (more plan steps, slower planner)

Smaller zones mean more precise robot positioning instructions.

```yaml
planner:
  grid_size: 8          # 8x8 = 64 zones
  replan_interval: 150  # replan more often to track fine movement
```

> **Warning**: `GRID_SIZE > 8` will cause the `behind_flock` predicate table
> to grow as O(N⁴) and may slow down problem generation noticeably.

### Example D — Baseline comparison (reactive-only)

Run both experiments and compare the logs:

```bash
python3 experiments/run_experiment.py          # saves data/logs/run.json
python3 experiments/run_pddl_experiment.py     # saves data/logs/run_pddl.json
```

Then visualise both:

```bash
python3 experiments/plot_run.py                # edit the script to point at each log
```

### Example E — Always run the same experiment (reproducible)

Use a fixed seed and fixed start positions.

```yaml
simulation:
  seed: 42
  robot_start_mode: fixed
  robot_start_positions: [[0.0, 0.0]]
  sheep_start_mode: fixed
  sheep_start_positions:
    - [1.0, 1.0]
    - [2.0, 1.0]
    - [3.0, 1.0]
    - [4.0, 1.0]
    - [5.0, 1.0]
    - [1.0, 2.0]
    - [2.0, 2.0]
    - [3.0, 2.0]
    - [4.0, 2.0]
    - [5.0, 2.0]
    - [1.0, 3.0]
    - [2.0, 3.0]
    - [3.0, 3.0]
    - [4.0, 3.0]
    - [5.0, 3.0]

flocking:
  noise_std: 0.01  # stochastic but reproducible with fixed seed
```

Or use the provided ready-to-run file:

```bash
EXPERIMENT_CONFIG=config/fixed_experiment.yaml \
python3 experiments/run_pddl_experiment.py
```

### Example F — Different run every execution

Use `seed: null` and random robot start.

```yaml
simulation:
  seed: null
  robot_start_mode: random

flocking:
  noise_std: 0.01
```

### Example G — Multi-robot fixed starts

```yaml
simulation:
  num_robots: 3
  robot_start_mode: fixed
  robot_start_positions:
    - [0.0, 0.0]
    - [2.0, 1.0]
    - [5.0, 5.0]
```

If you provide fewer positions than `num_robots`, the first position is reused.

---

## 5. Visualising Results

### Static plot

Edit `experiments/plot_run.py` to set the log path, then:

```bash
python3 experiments/plot_run.py
```

### Video export

```bash
python3 experiments/plot_run_video.py
```

Output is saved as an `.mp4` file. The `FRAME_STEP` and `FPS` constants in
that script control playback speed.

---

## 6. Inspecting the Generated PDDL Problem

After running the experiment once, the last auto-generated problem file is
available at `pddl/problem.pddl`. You can inspect it to understand the
current symbolic state:

```bash
cat pddl/problem.pddl
```

You can also run the planner manually against it:

```bash
python3 -m pyperplan -s astar -H hadd \
    pddl/shepherding_domain.pddl \
    pddl/problem.pddl
```

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: pyperplan is not installed` | Missing dependency | `pip install pyperplan` |
| `[Planner] Plan length: 0 actions` | Problem is unsolvable from current state | Increase `GRID_SIZE` or lower `collect_threshold` so the abstract state is less coarse |
| Simulation never reaches goal | Flock overshoots or oscillates | Reduce `robot_max_speed` or increase `REPLAN_INTERVAL` |
| Very slow startup | Large `behind_flock` predicate table | Reduce `GRID_SIZE` |
| `ModuleNotFoundError` for `shepherding.*` | Running from wrong directory | Always run scripts from the project root: `python3 experiments/run_pddl_experiment.py` |
