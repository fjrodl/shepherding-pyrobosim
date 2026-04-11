import json
import matplotlib.pyplot as plt
import numpy as np

fence = [(5,5), (15,5)]
goal = np.array([18, 18])


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


with open("data/logs/run.json") as f:
    data = json.load(f)

fig = plt.figure()
stop_requested = False


def _on_key_press(event):
    global stop_requested
    if event.key and event.key.lower() == "q":
        stop_requested = True
        plt.close(fig)


fig.canvas.mpl_connect("key_press_event", _on_key_press)

plt.ion()

for step in data[::10]:  # cada 10 pasos
    if stop_requested or not plt.fignum_exists(fig.number):
        break

    plt.clf()
    
    sheep = np.array(step["sheep"])
    robot = np.array(step["robot"])
    
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
    # 🤖 robot
    plt.scatter(robot[0], robot[1], marker='x', label="Robot")
    
    # 🎯 objetivo
    plt.scatter(goal[0], goal[1], marker='s', label="Goal")
    
    # 🚧 valla
    x_vals = [fence[0][0], fence[1][0]]
    y_vals = [fence[0][1], fence[1][1]]
    plt.plot(x_vals, y_vals)
    
    # 📍 centro del rebaño
    center = sheep.mean(axis=0)
    plt.scatter(center[0], center[1], marker='o', label="Center")
    
    # 🧠 título con modo
    plt.title(f"Step {step['time']} - Mode: {step.get('mode', '')}")
    
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.legend()
    
    plt.pause(0.05)

plt.ioff()
if not stop_requested and plt.fignum_exists(fig.number):
    plt.show()