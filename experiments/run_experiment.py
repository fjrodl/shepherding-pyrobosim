import numpy as np
import os
import sys
import yaml
import json

# Make sure the project root is on the path when running from experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shepherding.sheep import Sheep
from shepherding.robot import ShepherdRobot
from sim_logging.logger import Logger

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "pddl_experiment.yaml")


def _load_baseline_config(config_path):
    """Load reactive-only experiment settings from YAML with safe defaults."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    sim_cfg = raw.get("simulation", {})
    metrics_cfg = raw.get("metrics", {})
    flocking_cfg = raw.get("flocking", {})
    robot_cfg = raw.get("robot", {})

    cfg = {
        "seed": int(sim_cfg.get("seed", 42)),
        "num_robots": int(sim_cfg.get("num_robots", 1)),
        "robot_start_mode": str(sim_cfg.get("robot_start_mode", "origin")).lower(),
        "robot_start_positions": sim_cfg.get("robot_start_positions", [[0.0, 0.0]]),
        "num_sheep": int(sim_cfg.get("num_sheep", 15)),
        # null in YAML means run until goal reached
        "steps": sim_cfg.get("steps", None),
        "bounds": sim_cfg.get("bounds", [0, 0, 20, 20]),
        "goal_pos": np.array(sim_cfg.get("goal_pos", [18.0, 18.0]), dtype=float),
        "goal_radius": float(sim_cfg.get("goal_radius", 2.0)),
        "metrics_enabled": bool(metrics_cfg.get("enabled", True)),
        "metrics_output_path": metrics_cfg.get("reactive_output_path", "data/logs/run_metrics.json"),
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
    cfg["params"]["bounds"] = cfg["bounds"]
    return cfg


CFG = _load_baseline_config(CONFIG_PATH)

SEED = CFG["seed"]
NUM_ROBOTS = CFG["num_robots"]
ROBOT_START_MODE = CFG["robot_start_mode"]
ROBOT_START_POSITIONS = CFG["robot_start_positions"]
NUM_SHEEP = CFG["num_sheep"]
STEPS = CFG["steps"]
GOAL_POS = CFG["goal_pos"]
GOAL_RADIUS = CFG["goal_radius"]
METRICS_ENABLED = CFG["metrics_enabled"]
METRICS_OUTPUT_PATH = CFG["metrics_output_path"]
params = CFG["params"]


def _with_index_suffix(path, idx, total):
    if total <= 1:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_robot{idx}{ext}"


def _get_robot_start_position(robot_idx):
    if ROBOT_START_MODE == "random":
        xmin, ymin, xmax, ymax = params["bounds"]
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
    np.random.seed(SEED + robot_idx)

    sheep = [Sheep(i, np.random.rand(2) * 10) for i in range(NUM_SHEEP)]
    robot_start = _get_robot_start_position(robot_idx)
    robot = ShepherdRobot(position=robot_start)
    logger = Logger(path=_with_index_suffix("data/logs/run.json", robot_idx, NUM_ROBOTS))

    fence = [(5, 5), (15, 5)]

    t = 0
    goal_reached = False
    while True:
        if STEPS is not None and t >= STEPS:
            print(f"[Sim] Reached step limit ({STEPS}) without reaching goal.")
            break

        for s in sheep:
            neighbors = [n for n in sheep if n != s]
            s.update(neighbors, robot, params, fence)

        mode = robot.compute_action(sheep, GOAL_POS, params, fence=fence)
        logger.log(t, sheep, robot, mode=mode)

        centroid = np.mean([s.position for s in sheep], axis=0)
        if np.linalg.norm(centroid - GOAL_POS) <= GOAL_RADIUS:
            print(f"[Sim] Goal reached at step {t}!")
            goal_reached = True
            break

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


for robot_idx in range(NUM_ROBOTS):
    run_single_robot(robot_idx)

print("[Sim] Done.")