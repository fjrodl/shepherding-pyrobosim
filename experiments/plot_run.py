import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.environ.get(
    "EXPERIMENT_CONFIG",
    os.path.join(PROJECT_ROOT, "config", "pddl_experiment.yaml"),
)


def load_plot_config():
    """Load plot settings from the active experiment config."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file) or {}
    except FileNotFoundError:
        raw = {}

    sim_cfg = raw.get("simulation", {})
    return {
        "fence": sim_cfg.get("fence", [[5.0, 5.0], [15.0, 5.0]]),
        "goal": np.array(sim_cfg.get("goal_pos", [18.0, 18.0]), dtype=float),
        "bounds": sim_cfg.get("bounds", [0.0, 0.0, 20.0, 20.0]),
    }


PLOT_CONFIG = load_plot_config()
fence = PLOT_CONFIG["fence"]
goal = PLOT_CONFIG["goal"]
bounds = PLOT_CONFIG["bounds"]


def get_latest_log_file():
    """Find the most recently updated trajectory log file."""
    log_dir = "data/logs"
    candidate_files = [
        os.path.join(log_dir, "run.json"),
        os.path.join(log_dir, "run_pddl.json"),
    ]
    existing_files = [path for path in candidate_files if os.path.exists(path)]
    if not existing_files:
        return None
    return max(existing_files, key=os.path.getmtime)


def compute_clusters(positions, radius):
    clusters = []
    unvisited = set(range(len(positions)))

    while unvisited:
        i = unvisited.pop()
        cluster = [i]
        to_check = [i]

        while to_check:
            j = to_check.pop()
            for k in list(unvisited):
                if np.linalg.norm(positions[j] - positions[k]) < radius:
                    unvisited.remove(k)
                    cluster.append(k)
                    to_check.append(k)

        clusters.append(cluster)

    return clusters


fig = plt.figure()
stop_requested = False


def _on_key_press(event):
    global stop_requested
    if event.key and event.key.lower() == "q":
        stop_requested = True
        plt.close(fig)


def load_data():
    """Load data from the latest JSON log file with error handling."""
    log_file = get_latest_log_file()
    if not log_file:
        return []
    try:
        with open(log_file) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


fig.canvas.mpl_connect("key_press_event", _on_key_press)

plt.ion()

frames_shown = 0
while True:
    if stop_requested or not plt.fignum_exists(fig.number):
        break
    
    data = load_data()
    if not data:
        plt.pause(0.1)  # Wait for data to appear
        continue
    
    # Only process new frames (every 10 steps)
    for step in data[frames_shown::10]:
        if stop_requested or not plt.fignum_exists(fig.number):
            break

        plt.clf()
        
        sheep = np.array(step["sheep"])
        if "robots" in step:
            robots = np.array(step["robots"], dtype=float)
        elif "robot" in step:
            robots = np.array([step["robot"]], dtype=float)
        else:
            robots = np.empty((0, 2), dtype=float)
        
        # 🐑 ovejas
        # plt.scatter(sheep[:,0], sheep[:,1], label="Sheep")
        positions = np.array(step["sheep"])
        clusters = compute_clusters(positions, radius=2.5)

        colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta']

        for i, cluster in enumerate(clusters):
            pts = positions[cluster]
            plt.scatter(
                pts[:,0], pts[:,1],
                color=colors[i % len(colors)],
                label=f"Cluster {i}" if step == data[0] else ""
            )
        # 🤖 robot(s)
        if len(robots) > 0:
            plt.scatter(robots[:,0], robots[:,1], marker='x', s=200, c='red', linewidths=2.5, label="Robot", zorder=5)
        
        # 🎯 objetivo
        plt.scatter(goal[0], goal[1], marker='s', label="Goal")
        
        # 🚧 valla
        if fence is not None:
            x_vals = [fence[0][0], fence[1][0]]
            y_vals = [fence[0][1], fence[1][1]]
            plt.plot(x_vals, y_vals)
        
        # 📍 centro del rebaño
        center = sheep.mean(axis=0)
        plt.scatter(center[0], center[1], marker='o', label="Center")
        
        # 🧠 título con modo
        plt.title(f"Step {step['time']} - Mode: {step.get('mode', '')}")
        
        plt.xlim(bounds[0], bounds[2])
        plt.ylim(bounds[1], bounds[3])
        plt.legend()
        
        plt.pause(0.05)
    
    # Update frames shown and check if simulation is done
    frames_shown = len(data)
    if frames_shown > 0 and len(data[frames_shown-1:frames_shown]) == 0:
        # Data stopped growing, likely simulation complete
        plt.pause(0.5)
    else:
        plt.pause(0.1)

plt.ioff()
if not stop_requested and plt.fignum_exists(fig.number):
    plt.show()