import json
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os

# ----------------------------
# CONFIG
# ----------------------------
fence = [(5,5), (15,5)]
goal = np.array([18, 18])
CLUSTER_RADIUS = 2.5
FRAME_STEP = 10   # usa data[::FRAME_STEP]
FPS = 10

# ----------------------------
# CLUSTERING
# ----------------------------
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

# ----------------------------
# LOAD DATA
# ----------------------------
with open("data/logs/run.json") as f:
    data = json.load(f)

# ----------------------------
# PREPARE FRAMES DIR
# ----------------------------
os.makedirs("frames", exist_ok=True)

plt.ioff()  # importante para guardar frames

# ----------------------------
# GENERATE FRAMES
# ----------------------------
for idx, step in enumerate(data[::FRAME_STEP]):
    plt.clf()

    positions = np.array(step["sheep"])
    # Support both legacy "robot" (single position) and current "robots" (list).
    if "robots" in step:
        robots = np.array(step["robots"], dtype=float)
    elif "robot" in step:
        robots = np.array([step["robot"]], dtype=float)
    else:
        raise KeyError("Missing robot data: expected 'robots' or 'robot' in log step")

    # --- clusters ---
    clusters = compute_clusters(positions, CLUSTER_RADIUS)
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta']

    for i, cluster in enumerate(clusters):
        pts = positions[cluster]
        plt.scatter(
            pts[:,0], pts[:,1],
            color=colors[i % len(colors)],
            s=20
        )

    # --- robot(s) ---
    if len(robots) > 0:
        plt.scatter(robots[:, 0], robots[:, 1], marker='x', s=80, label="Robot")

    # --- goal ---
    plt.scatter(goal[0], goal[1], marker='s', s=80, label="Goal")

    # --- fence ---
    x_vals = [fence[0][0], fence[1][0]]
    y_vals = [fence[0][1], fence[1][1]]
    # plt.plot(x_vals, y_vals, linewidth=3, color='brown', label="Fence")

    # --- center ---
    center = positions.mean(axis=0)
    plt.scatter(center[0], center[1], marker='o', s=60, label="Center")

    # --- title ---
    plt.title(
        f"Step {step['time']} | Mode: {step.get('mode','')} | Clusters: {len(clusters)}"
    )

    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.legend(loc="upper left")

    # --- save frame ---
    frame_path = f"frames/frame_{idx:04d}.png"
    plt.savefig(frame_path)

print("Frames generados.")

# ----------------------------
# CREATE VIDEO
# ----------------------------
frames = []
frame_files = sorted(os.listdir("frames"))

for file in frame_files:
    if file.endswith(".png"):
        frames.append(imageio.imread(os.path.join("frames", file)))

imageio.mimsave("simulation.mp4", frames, fps=FPS)

print("🎬 Video guardado como simulation.mp4")