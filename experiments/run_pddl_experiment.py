"""
run_pddl_experiment.py
----------------------
Main loop for the PDDL-guided shepherding experiment.

Architecture
------------
  Deliberative layer  — WorldAbstraction + PDDL planner (pyperplan)
  Reactive layer      — ShepherdRobot.compute_action() + Sheep.update()

The planner is called:
  1. At step 0 (initial plan).
  2. Every REPLAN_INTERVAL steps (periodic re-planning).
  3. When the current plan action is completed (advance plan or replan).
  4. When the plan becomes empty before the goal is reached (replan).
"""

import numpy as np
import os
import sys
import yaml
import json

# Make sure the project root is on the path when running from experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shepherding.sheep import Sheep
from shepherding.robot import ShepherdRobot
from shepherding.world_abstraction import WorldAbstraction
from shepherding.planner_interface import solve
from shepherding.plan_executor import PlanExecutor
from pddl.problem_generator import generate_problem
from sim_logging.logger import Logger

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "pddl_experiment.yaml")


def _load_experiment_config(config_path):
    """Load experiment configuration from YAML and apply safe defaults."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    sim_cfg = raw.get("simulation", {})
    planner_cfg = raw.get("planner", {})
    logging_cfg = raw.get("logging", {})
    metrics_cfg = raw.get("metrics", {})
    pddl_cfg = raw.get("pddl", {})
    flocking_cfg = raw.get("flocking", {})
    robot_cfg = raw.get("robot", {})

    cfg = {
        "seed": int(sim_cfg.get("seed", 42)),
        "num_robots": int(sim_cfg.get("num_robots", 1)),
        "robot_start_mode": str(sim_cfg.get("robot_start_mode", "origin")).lower(),
        "robot_start_positions": sim_cfg.get("robot_start_positions", [[0.0, 0.0]]),
        "num_sheep": int(sim_cfg.get("num_sheep", 15)),
        # YAML null means unlimited steps (run until goal reached)
        "steps": sim_cfg.get("steps", None),
        "bounds": sim_cfg.get("bounds", [0, 0, 20, 20]),
        "goal_pos": sim_cfg.get("goal_pos", [18.0, 18.0]),
        "goal_radius": float(sim_cfg.get("goal_radius", 2.0)),
        "replan_interval": int(planner_cfg.get("replan_interval", 300)),
        "grid_size": int(planner_cfg.get("grid_size", 5)),
        "planner_backend": planner_cfg.get("backend", "pyperplan"),
        "popf_command": planner_cfg.get("popf_command", "popf"),
        "planner_timeout_s": planner_cfg.get("timeout_s", None),
        "planner_quiet": bool(logging_cfg.get("planner_quiet", True)),
        "iteration_log_interval": int(logging_cfg.get("iteration_log_interval", 100)),
        "metrics_enabled": bool(metrics_cfg.get("enabled", True)),
        "metrics_output_path": metrics_cfg.get("pddl_output_path", "data/logs/run_pddl_metrics.json"),
        "domain_path": os.path.join(
            PROJECT_ROOT,
            pddl_cfg.get("domain_path", "pddl/shepherding_domain.pddl"),
        ),
        "problem_path": os.path.join(
            PROJECT_ROOT,
            pddl_cfg.get("problem_path", "pddl/problem.pddl"),
        ),
        "problem_name": pddl_cfg.get("problem_name", "shepherding-problem"),
        "domain_name": pddl_cfg.get("domain_name", "shepherding"),
        "params": {
            "w_coh": float(flocking_cfg.get("w_coh", 0.05)),
            "w_sep": float(flocking_cfg.get("w_sep", 0.4)),
            "w_align": float(flocking_cfg.get("w_align", 0.05)),
            "w_robot": float(flocking_cfg.get("w_robot", 0.2)),
            "w_fence": float(flocking_cfg.get("w_fence", 1.5)),
            "min_dist": float(flocking_cfg.get("min_dist", 0.5)),
            "max_speed": float(flocking_cfg.get("max_speed", 0.15)),
            "dt": float(flocking_cfg.get("dt", 0.05)),
            "neighbor_radius": float(flocking_cfg.get("neighbor_radius", 2.5)),
            "robot_influence": float(robot_cfg.get("robot_influence", 3.0)),
            "collect_threshold": float(robot_cfg.get("collect_threshold", 3.0)),
            "collect_distance": float(robot_cfg.get("collect_distance", 1.5)),
            "drive_distance": float(robot_cfg.get("drive_distance", 2.0)),
            "robot_max_speed": float(robot_cfg.get("robot_max_speed", 0.3)),
            "robot_sheep_min_distance": float(robot_cfg.get("robot_sheep_min_distance", 1.0)),
            "robot_fence_clearance": float(robot_cfg.get("robot_fence_clearance", 0.35)),
        },
    }
    return cfg


CFG = _load_experiment_config(CONFIG_PATH)

SEED = CFG["seed"]
NUM_ROBOTS = CFG["num_robots"]
ROBOT_START_MODE = CFG["robot_start_mode"]
ROBOT_START_POSITIONS = CFG["robot_start_positions"]
NUM_SHEEP = CFG["num_sheep"]
STEPS = CFG["steps"]
REPLAN_INTERVAL = CFG["replan_interval"]
GRID_SIZE = CFG["grid_size"]
GOAL_RADIUS = CFG["goal_radius"]
BOUNDS = CFG["bounds"]
GOAL_POS = np.array(CFG["goal_pos"], dtype=float)
PLANNER_BACKEND = CFG["planner_backend"]
POPF_COMMAND = CFG["popf_command"]
PLANNER_TIMEOUT_S = CFG["planner_timeout_s"]
PLANNER_QUIET = CFG["planner_quiet"]
ITERATION_LOG_INTERVAL = CFG["iteration_log_interval"]
METRICS_ENABLED = CFG["metrics_enabled"]
METRICS_OUTPUT_PATH = CFG["metrics_output_path"]

DOMAIN_PATH = CFG["domain_path"]
PROBLEM_PATH = CFG["problem_path"]
PROBLEM_NAME = CFG["problem_name"]
DOMAIN_NAME = CFG["domain_name"]

params = CFG["params"]
params["bounds"] = BOUNDS

fence = [(5, 5), (15, 5)]

def _with_index_suffix(path, idx, total):
    """Append _robotN suffix to a path when running multi-robot batches."""
    if total <= 1:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_robot{idx}{ext}"


def _get_robot_start_position(robot_idx):
    if ROBOT_START_MODE == "random":
        xmin, ymin, xmax, ymax = BOUNDS
        return np.array([
            np.random.uniform(xmin, xmax),
            np.random.uniform(ymin, ymax),
        ], dtype=float)

    if ROBOT_START_MODE == "fixed":
        if not ROBOT_START_POSITIONS:
            raise ValueError("robot_start_positions must be provided when robot_start_mode=fixed")
        idx = robot_idx if robot_idx < len(ROBOT_START_POSITIONS) else 0
        return np.array(ROBOT_START_POSITIONS[idx], dtype=float)

    # default: origin
    return np.array([0.0, 0.0], dtype=float)


def run_single_robot(robot_idx):
    """Run one independent robot experiment instance."""
    np.random.seed(SEED + robot_idx)
    sheep = [Sheep(i, np.random.rand(2) * 10) for i in range(NUM_SHEEP)]
    robot_start = _get_robot_start_position(robot_idx)
    robot = ShepherdRobot(position=robot_start)

    log_path = _with_index_suffix("data/logs/run_pddl.json", robot_idx, NUM_ROBOTS)
    problem_path = _with_index_suffix(PROBLEM_PATH, robot_idx, NUM_ROBOTS)
    problem_name = f"{PROBLEM_NAME}-robot{robot_idx}" if NUM_ROBOTS > 1 else PROBLEM_NAME

    logger = Logger(path=log_path)

    abstraction = WorldAbstraction(
        bounds=BOUNDS,
        grid_size=GRID_SIZE,
        collect_threshold=params["collect_threshold"],
    )
    executor = PlanExecutor(abstraction, goal_reached_radius=GOAL_RADIUS)

    print(
        f"[Sim] Starting robot {robot_idx + 1}/{NUM_ROBOTS} "
        f"(planner={PLANNER_BACKEND})."
    )

    def replan(local_sheep, local_robot, step_idx):
        """Compute a new abstract state, generate a problem file and solve it."""
        print(f"[Sim] Iteration {step_idx}: re-planning ...", flush=True)
        state = abstraction.compute(local_sheep, local_robot, GOAL_POS)
        generate_problem(
            state,
            domain_name=DOMAIN_NAME,
            problem_name=problem_name,
            output_path=problem_path,
        )
        try:
            new_plan = solve(
                os.path.abspath(DOMAIN_PATH),
                os.path.abspath(problem_path),
                backend=PLANNER_BACKEND,
                popf_command=POPF_COMMAND,
                timeout_s=PLANNER_TIMEOUT_S,
            )
        except RuntimeError as exc:
            print(f"[Planner] WARNING: {exc}")
            new_plan = []
        if not PLANNER_QUIET:
            print(f"[Planner] Plan length: {len(new_plan)} actions")
            for i, a in enumerate(new_plan):
                print(f"  {i+1}. {a['name']} {' '.join(a['args'])}")
        return new_plan

    plan = replan(sheep, robot, step_idx=0)
    current_action = plan.pop(0) if plan else None
    steps_since_replan = 0

    t = 0
    goal_reached = False
    while True:
        if STEPS is not None and t >= STEPS:
            print(f"[Sim] Reached step limit ({STEPS}) without reaching goal.")
            break

        if ITERATION_LOG_INTERVAL > 0 and t % ITERATION_LOG_INTERVAL == 0:
            print(f"[Sim] Iteration {t}")

        # ---- Reactive layer: advance simulation one step ----
        for s in sheep:
            neighbors = [n for n in sheep if n != s]
            s.update(neighbors, robot, params, fence)

        # Determine effective params from current PDDL action
        eff_params = executor.get_param_overrides(current_action, sheep, robot, GOAL_POS, params)
        mode = robot.compute_action(sheep, GOAL_POS, eff_params, fence=fence)

        # Override logged mode with the PDDL action name for clarity
        log_mode = current_action["name"] if current_action else "idle"
        logger.log(t, sheep, robot, mode=log_mode)

        # ---- Check mission complete ----
        centroid = np.mean([s.position for s in sheep], axis=0)
        if np.linalg.norm(centroid - GOAL_POS) <= GOAL_RADIUS:
            print(f"[Sim] Goal reached at step {t}!")
            goal_reached = True
            break

        # ---- Advance plan if current action is done ----
        steps_since_replan += 1
        if current_action is not None and executor.action_completed(
            current_action, sheep, robot, GOAL_POS
        ):
            print(f"[Sim] Step {t}: action '{current_action['name']}' completed.")
            if plan:
                current_action = plan.pop(0)
                print(f"[Sim] Next action: {current_action['name']} {current_action['args']}")
            else:
                # Plan exhausted — replan
                plan = replan(sheep, robot, step_idx=t)
                current_action = plan.pop(0) if plan else None
                steps_since_replan = 0

        # ---- Periodic re-planning ----
        elif steps_since_replan >= REPLAN_INTERVAL:
            plan = replan(sheep, robot, step_idx=t)
            current_action = plan.pop(0) if plan else None
            steps_since_replan = 0

        t += 1

    logger.save()
    if METRICS_ENABLED:
        centroid = np.mean([s.position for s in sheep], axis=0)
        sheep_positions = np.array([s.position for s in sheep])
        sheep_goal_dists = np.linalg.norm(sheep_positions - GOAL_POS, axis=1)
        robot_sheep_dists = np.linalg.norm(sheep_positions - robot.position, axis=1)
        metrics = {
            "robot_index": robot_idx,
            "steps_executed": t,
            "goal_reached": goal_reached,
            "planner_backend": PLANNER_BACKEND,
            "final_centroid_distance_to_goal": float(np.linalg.norm(centroid - GOAL_POS)),
            "avg_sheep_distance_to_goal": float(np.mean(sheep_goal_dists)),
            "max_sheep_distance_to_goal": float(np.max(sheep_goal_dists)),
            "min_robot_sheep_distance": float(np.min(robot_sheep_dists)),
            "avg_robot_sheep_distance": float(np.mean(robot_sheep_dists)),
            "simulated_time": float(t * params["dt"]),
        }

        metrics_path = _with_index_suffix(METRICS_OUTPUT_PATH, robot_idx, NUM_ROBOTS)
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Metrics] Saved metrics to {metrics_path}")

    print(f"[Sim] Robot {robot_idx + 1}/{NUM_ROBOTS} done.")


for robot_idx in range(NUM_ROBOTS):
    run_single_robot(robot_idx)

print("[Sim] Done.")
