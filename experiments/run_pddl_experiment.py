"""
run_pddl_experiment.py
----------------------
Main loop for the PDDL-guided shepherding experiment.

Architecture
------------
  Deliberative layer  — WorldAbstraction + PDDL planner (pyperplan/POPF)
  Reactive layer      — ShepherdRobot.compute_action() + Sheep.update()

In multi-robot mode, all robots share the same arena and flock. Each robot
maintains its own PDDL planning state, but replans from the same shared world.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pddl.problem_generator import generate_problem
from shepherding.plan_executor import PlanExecutor
from shepherding.planner_interface import solve
from shepherding.robot import ShepherdRobot
from shepherding.sheep import Sheep
from shepherding.world_abstraction import WorldAbstraction
from sim_logging.logger import Logger


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.environ.get(
    "EXPERIMENT_CONFIG",
    os.path.join(PROJECT_ROOT, "config", "pddl_experiment.yaml"),
)


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

    seed_raw = sim_cfg.get("seed", 42)
    seed = None if seed_raw is None else int(seed_raw)

    cfg = {
        "seed": seed,
        "num_robots": int(sim_cfg.get("num_robots", 1)),
        "coordination_mode": str(sim_cfg.get("coordination_mode", "roles")).lower(),
        "robot_start_mode": str(sim_cfg.get("robot_start_mode", "origin")).lower(),
        "robot_start_positions": sim_cfg.get("robot_start_positions", [[0.0, 0.0]]),
        "sheep_start_mode": str(sim_cfg.get("sheep_start_mode", "random")).lower(),
        "sheep_start_positions": sim_cfg.get("sheep_start_positions", []),
        "sheep_spawn_bounds": sim_cfg.get("sheep_spawn_bounds", [0.0, 0.0, 10.0, 10.0]),
        "num_sheep": int(sim_cfg.get("num_sheep", 15)),
        "steps": sim_cfg.get("steps", None),
        "bounds": sim_cfg.get("bounds", [0, 0, 20, 20]),
        "goal_pos": np.array(sim_cfg.get("goal_pos", [18.0, 18.0]), dtype=float),
        "goal_radius": float(sim_cfg.get("goal_radius", 2.0)),
        "fence": sim_cfg.get("fence", None),
        "replan_interval": int(planner_cfg.get("replan_interval", 300)),
        "grid_size": int(planner_cfg.get("grid_size", 5)),
        "planner_backend": str(planner_cfg.get("backend", "pyperplan")).lower(),
        "planner_fallback_backend": (
            str(planner_cfg.get("fallback_backend", "")).lower() or None
        ),
        "popf_command": planner_cfg.get("popf_command", "popf"),
        "planner_timeout_s": planner_cfg.get("timeout_s", None),
        "planner_quiet": bool(logging_cfg.get("planner_quiet", True)),
        "iteration_log_interval": int(logging_cfg.get("iteration_log_interval", 100)),
        "visualization_enabled": bool(logging_cfg.get("enable_visualization", False)),
        "metrics_enabled": bool(metrics_cfg.get("enabled", True)),
        "metrics_output_path": metrics_cfg.get("pddl_output_path", "data/logs/run_pddl_metrics.json"),
        "domain_path": os.path.join(PROJECT_ROOT, pddl_cfg.get("domain_path", "pddl/shepherding_domain.pddl")),
        "problem_path": os.path.join(PROJECT_ROOT, pddl_cfg.get("problem_path", "pddl/problem.pddl")),
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
            "noise_std": float(flocking_cfg.get("noise_std", 0.01)),
            "robot_influence": float(robot_cfg.get("robot_influence", 3.0)),
            "collect_threshold": float(robot_cfg.get("collect_threshold", 3.0)),
            "collect_distance": float(robot_cfg.get("collect_distance", 1.5)),
            "drive_distance": float(robot_cfg.get("drive_distance", 2.0)),
            "robot_max_speed": float(robot_cfg.get("robot_max_speed", 0.3)),
            "robot_sheep_min_distance": float(robot_cfg.get("robot_sheep_min_distance", 1.0)),
            "robot_fence_clearance": float(robot_cfg.get("robot_fence_clearance", 0.35)),
        },
    }
    cfg["params"]["bounds"] = cfg["bounds"]
    return cfg


CFG = _load_experiment_config(CONFIG_PATH)

SEED = CFG["seed"]
NUM_ROBOTS = CFG["num_robots"]
COORDINATION_MODE = CFG["coordination_mode"]
ROBOT_START_MODE = CFG["robot_start_mode"]
ROBOT_START_POSITIONS = CFG["robot_start_positions"]
SHEEP_START_MODE = CFG["sheep_start_mode"]
SHEEP_START_POSITIONS = CFG["sheep_start_positions"]
SHEEP_SPAWN_BOUNDS = CFG["sheep_spawn_bounds"]
NUM_SHEEP = CFG["num_sheep"]
STEPS = CFG["steps"]
REPLAN_INTERVAL = CFG["replan_interval"]
GRID_SIZE = CFG["grid_size"]
GOAL_RADIUS = CFG["goal_radius"]
BOUNDS = CFG["bounds"]
GOAL_POS = CFG["goal_pos"]
FENCE = CFG["fence"]
PLANNER_BACKEND = CFG["planner_backend"]
PLANNER_FALLBACK_BACKEND = CFG["planner_fallback_backend"]
POPF_COMMAND = CFG["popf_command"]
PLANNER_TIMEOUT_S = CFG["planner_timeout_s"]
PLANNER_QUIET = CFG["planner_quiet"]
ITERATION_LOG_INTERVAL = CFG["iteration_log_interval"]
VISUALIZATION_ENABLED = CFG["visualization_enabled"]
METRICS_ENABLED = CFG["metrics_enabled"]
METRICS_OUTPUT_PATH = CFG["metrics_output_path"]
DOMAIN_PATH = CFG["domain_path"]
PROBLEM_PATH = CFG["problem_path"]
PROBLEM_NAME = CFG["problem_name"]
DOMAIN_NAME = CFG["domain_name"]
params = CFG["params"]
fence = FENCE


def _with_index_suffix(path, idx, total):
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
        if robot_idx < len(ROBOT_START_POSITIONS):
            return np.array(ROBOT_START_POSITIONS[robot_idx], dtype=float)
        # Fewer positions than robots: offset each extra robot by 0.5 units along x
        base = np.array(ROBOT_START_POSITIONS[0], dtype=float)
        offset = (robot_idx - len(ROBOT_START_POSITIONS) + 1) * 0.5
        return base + np.array([offset, 0.0], dtype=float)

    return np.array([0.0, 0.0], dtype=float)


def _get_sheep_start_positions():
    if SHEEP_START_MODE == "fixed":
        if len(SHEEP_START_POSITIONS) < NUM_SHEEP:
            raise ValueError(
                "sheep_start_positions must contain at least num_sheep entries when sheep_start_mode=fixed"
            )
        return [np.array(SHEEP_START_POSITIONS[i], dtype=float) for i in range(NUM_SHEEP)]

    xmin, ymin, xmax, ymax = SHEEP_SPAWN_BOUNDS
    return [
        np.array([
            np.random.uniform(xmin, xmax),
            np.random.uniform(ymin, ymax),
        ], dtype=float)
        for _ in range(NUM_SHEEP)
    ]


def run_shared_pddl_simulation():
    if COORDINATION_MODE not in {"roles", "pddl_all_robots"}:
        raise ValueError(
            "simulation.coordination_mode must be 'roles' or 'pddl_all_robots'"
        )

    selected_backend = PLANNER_BACKEND
    if selected_backend == "popf":
        cmd_parts = shlex.split(str(POPF_COMMAND))
        popf_bin = cmd_parts[0] if cmd_parts else "popf"
        if shutil.which(popf_bin) is None:
            if PLANNER_FALLBACK_BACKEND in {"pyperplan", "popf"} and PLANNER_FALLBACK_BACKEND != "popf":
                print(
                    f"[Planner] WARNING: POPF executable '{popf_bin}' not found. "
                    f"Falling back to backend='{PLANNER_FALLBACK_BACKEND}'."
                )
                selected_backend = PLANNER_FALLBACK_BACKEND
            else:
                raise RuntimeError(
                    f"POPF executable '{popf_bin}' not found on PATH. "
                    "Install POPF or set planner.fallback_backend: pyperplan in YAML."
                )

    if SEED is None:
        np.random.seed(None)
    else:
        np.random.seed(SEED)

    sheep_starts = _get_sheep_start_positions()
    sheep = [Sheep(i, sheep_starts[i]) for i in range(NUM_SHEEP)]
    robots = [
        ShepherdRobot(position=_get_robot_start_position(robot_idx))
        for robot_idx in range(NUM_ROBOTS)
    ]

    logger = Logger(path="data/logs/run.json")

    abstractions = [
        WorldAbstraction(bounds=BOUNDS, grid_size=GRID_SIZE, collect_threshold=params["collect_threshold"])
        for _ in range(NUM_ROBOTS)
    ]
    executors = [
        PlanExecutor(abstractions[robot_idx], goal_reached_radius=GOAL_RADIUS)
        for robot_idx in range(NUM_ROBOTS)
    ]
    problem_paths = [_with_index_suffix(PROBLEM_PATH, robot_idx, NUM_ROBOTS) for robot_idx in range(NUM_ROBOTS)]
    problem_names = [
        f"{PROBLEM_NAME}-robot{robot_idx}" if NUM_ROBOTS > 1 else PROBLEM_NAME
        for robot_idx in range(NUM_ROBOTS)
    ]

    print(
        f"[Sim] Starting shared-arena run with {NUM_ROBOTS} robots "
        f"(planner={selected_backend}, coordination_mode={COORDINATION_MODE})."
    )

    def replan(robot_idx, step_idx):
        abstraction = abstractions[robot_idx]
        robot = robots[robot_idx]
        print(f"[Sim] Iteration {step_idx}: re-planning for robot {robot_idx} ...", flush=True)
        state = abstraction.compute(sheep, robot, GOAL_POS)
        generate_problem(
            state,
            domain_name=DOMAIN_NAME,
            problem_name=problem_names[robot_idx],
            output_path=problem_paths[robot_idx],
        )
        try:
            new_plan = solve(
                os.path.abspath(DOMAIN_PATH),
                os.path.abspath(problem_paths[robot_idx]),
                backend=selected_backend,
                popf_command=POPF_COMMAND,
                timeout_s=PLANNER_TIMEOUT_S,
            )
        except RuntimeError as exc:
            print(f"[Planner] WARNING robot {robot_idx}: {exc}")
            new_plan = []

        if not PLANNER_QUIET:
            print(f"[Planner] Robot {robot_idx} plan length: {len(new_plan)} actions")
            for step_num, action in enumerate(new_plan, start=1):
                print(f"  {step_num}. {action['name']} {' '.join(action['args'])}")
        return new_plan

    def collect_score(robot_idx):
        positions = np.array([s.position for s in sheep])
        centroid = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        max_dist = float(np.max(distances))
        outlier_count = int(np.sum(distances > abstractions[robot_idx].collect_threshold))
        return max_dist, outlier_count

    if COORDINATION_MODE == "roles":
        # Robot 0 (alpha) is PDDL-guided; others are reactive roles.
        alpha_plan = replan(0, step_idx=0)
        plans = [alpha_plan] + [[] for _ in range(1, NUM_ROBOTS)]
        current_actions = [plans[0].pop(0) if plans[0] else None] + [None] * (NUM_ROBOTS - 1)
    else:
        # All robots receive their own PDDL plans.
        plans = [replan(robot_idx, step_idx=0) for robot_idx in range(NUM_ROBOTS)]
        current_actions = [plan.pop(0) if plan else None for plan in plans]
    steps_since_replan = [0 for _ in range(NUM_ROBOTS)]
    current_action_stall_steps = [0 for _ in range(NUM_ROBOTS)]
    best_collect_scores = [None for _ in range(NUM_ROBOTS)]

    def reset_action_progress(robot_idx):
        current_action_stall_steps[robot_idx] = 0
        action = current_actions[robot_idx]
        if action is not None and action["name"] == "collect-outlier":
            best_collect_scores[robot_idx] = collect_score(robot_idx)
        else:
            best_collect_scores[robot_idx] = None

    for robot_idx in range(NUM_ROBOTS):
        reset_action_progress(robot_idx)

    t = 0
    goal_reached = False
    plot_process = None
    flush_interval = 100

    while True:
        if STEPS is not None and t >= STEPS:
            print(f"[Sim] Reached step limit ({STEPS}) without reaching goal.")
            break

        if ITERATION_LOG_INTERVAL > 0 and t % ITERATION_LOG_INTERVAL == 0:
            print(f"[Sim] Iteration {t}")

        for s in sheep:
            neighbors = [n for n in sheep if n != s]
            s.update(neighbors, robots, params, fence)

        log_modes = []
        for robot_idx, robot in enumerate(robots):
            if COORDINATION_MODE == "roles":
                if robot_idx == 0:
                    # Alpha: PDDL-guided overrides
                    eff_params = executors[0].get_param_overrides(
                        current_actions[0], sheep, robot, GOAL_POS, params
                    )
                    eff_params = dict(eff_params)
                    role = "alpha"
                else:
                    # Subordinate robots: purely reactive, no PDDL
                    eff_params = dict(params)
                    role = "collector" if robot_idx == 1 else "flanker"
            else:
                # Per-robot PDDL mode
                eff_params = executors[robot_idx].get_param_overrides(
                    current_actions[robot_idx], sheep, robot, GOAL_POS, params
                )
                eff_params = dict(eff_params)
                role = "alpha"
            eff_params["robot_index"] = robot_idx
            eff_params["num_robots"] = NUM_ROBOTS
            eff_params["robot_role"] = role
            mode = robot.compute_action(sheep, GOAL_POS, eff_params, fence=fence)
            log_modes.append(current_actions[robot_idx]["name"] if current_actions[robot_idx] else mode)

        logger.log(t, sheep, robots, mode=log_modes)

        if t % flush_interval == 0:
            logger.flush()
            if VISUALIZATION_ENABLED and plot_process is None and t == flush_interval:
                try:
                    project_root = os.path.join(os.path.dirname(__file__), "..")
                    plot_process = subprocess.Popen(
                        [sys.executable, "experiments/plot_run.py"],
                        cwd=project_root,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    print("[Sim] Started live visualization (plot_run.py)")
                except Exception as exc:
                    print(f"[Sim] Could not start visualization: {exc}")

        centroid = np.mean([s.position for s in sheep], axis=0)
        if np.linalg.norm(centroid - GOAL_POS) <= GOAL_RADIUS:
            print(f"[Sim] Goal reached at step {t}!")
            goal_reached = True
            break

        for robot_idx in range(NUM_ROBOTS):
            if COORDINATION_MODE == "roles" and robot_idx != 0:
                continue  # subordinate robots are purely reactive; no PDDL tracking
            steps_since_replan[robot_idx] += 1
            action = current_actions[robot_idx]

            if action is not None and action["name"] == "collect-outlier":
                score = collect_score(robot_idx)
                improved = False
                if best_collect_scores[robot_idx] is None:
                    improved = True
                else:
                    best_max_dist, best_outlier_count = best_collect_scores[robot_idx]
                    max_dist, outlier_count = score
                    improved = (
                        outlier_count < best_outlier_count
                        or (outlier_count == best_outlier_count and max_dist < best_max_dist - 1e-3)
                    )

                if improved:
                    best_collect_scores[robot_idx] = score
                    current_action_stall_steps[robot_idx] = 0
                else:
                    current_action_stall_steps[robot_idx] += 1

            if action is not None and executors[robot_idx].action_completed(
                action, sheep, robots[robot_idx], GOAL_POS
            ):
                print(f"[Sim] Step {t}: robot {robot_idx} action '{action['name']}' completed.")
                if plans[robot_idx]:
                    current_actions[robot_idx] = plans[robot_idx].pop(0)
                    print(
                        f"[Sim] Robot {robot_idx} next action: "
                        f"{current_actions[robot_idx]['name']} {current_actions[robot_idx]['args']}"
                    )
                else:
                    plans[robot_idx] = replan(robot_idx, step_idx=t)
                    current_actions[robot_idx] = plans[robot_idx].pop(0) if plans[robot_idx] else None
                    steps_since_replan[robot_idx] = 0
                reset_action_progress(robot_idx)

            elif action is not None and action["name"] == "collect-outlier" and current_action_stall_steps[robot_idx] >= 75:
                print(
                    f"[Sim] Step {t}: robot {robot_idx} action '{action['name']}' stalled; forcing re-plan.",
                    flush=True,
                )
                plans[robot_idx] = replan(robot_idx, step_idx=t)
                current_actions[robot_idx] = plans[robot_idx].pop(0) if plans[robot_idx] else None
                steps_since_replan[robot_idx] = 0
                reset_action_progress(robot_idx)

            elif steps_since_replan[robot_idx] >= REPLAN_INTERVAL:
                plans[robot_idx] = replan(robot_idx, step_idx=t)
                current_actions[robot_idx] = plans[robot_idx].pop(0) if plans[robot_idx] else None
                steps_since_replan[robot_idx] = 0
                reset_action_progress(robot_idx)

        t += 1

    logger.save()

    if plot_process is not None:
        print("[Sim] Simulation complete. Visualization window will stay open until closed (press 'q' to quit).")
        try:
            plot_process.wait()
        except KeyboardInterrupt:
            plot_process.terminate()

    if METRICS_ENABLED:
        centroid = np.mean([s.position for s in sheep], axis=0)
        sheep_positions = np.array([s.position for s in sheep])
        sheep_goal_dists = np.linalg.norm(sheep_positions - GOAL_POS, axis=1)
        robot_positions = np.array([robot.position for robot in robots])
        robot_sheep_dist_matrix = np.linalg.norm(
            sheep_positions[:, None, :] - robot_positions[None, :, :], axis=2
        )
        nearest_robot_dists = np.min(robot_sheep_dist_matrix, axis=1)
        metrics = {
            "num_robots": NUM_ROBOTS,
            "coordination_mode": COORDINATION_MODE,
            "steps_executed": t,
            "goal_reached": goal_reached,
            "planner_backend": selected_backend,
            "final_centroid_distance_to_goal": float(np.linalg.norm(centroid - GOAL_POS)),
            "avg_sheep_distance_to_goal": float(np.mean(sheep_goal_dists)),
            "max_sheep_distance_to_goal": float(np.max(sheep_goal_dists)),
            "min_robot_sheep_distance": float(np.min(nearest_robot_dists)),
            "avg_robot_sheep_distance": float(np.mean(nearest_robot_dists)),
            "simulated_time": float(t * params["dt"]),
        }

        os.makedirs(os.path.dirname(METRICS_OUTPUT_PATH), exist_ok=True)
        with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Metrics] Saved metrics to {METRICS_OUTPUT_PATH}")


run_shared_pddl_simulation()

print("[Sim] Done.")
