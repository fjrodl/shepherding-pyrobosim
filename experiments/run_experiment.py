from shepherding.sheep import Sheep
from shepherding.robot import ShepherdRobot
import numpy as np
from sim_logging.logger import Logger

NUM_SHEEP = 15
STEPS = 15000

sheep = [Sheep(i, np.random.rand(2)*10) for i in range(NUM_SHEEP)]
robot = ShepherdRobot(position=[0,0])
goal = np.array([18,18])

logger = Logger()

params = {
    "w_coh": 0.05,
    "w_sep": 0.4,
    "w_align": 0.05,
    "w_robot": 0.2,
    "w_fence": 1.5,
    "min_dist": 0.5,
    "max_speed": 0.15,
    "dt": 0.05,
    "robot_influence": 3.0,
    "bounds": [0, 0, 20, 20],
    "collect_threshold": 3.0,     # cuándo el rebaño está disperso
    "collect_distance": 1.5,      # distancia detrás de oveja
    "drive_distance": 2.0,        # distancia detrás del grupo
    "robot_max_speed": 0.3,
    "neighbor_radius": 2.5,
}

fence = [(5,5), (15,5)]

for t in range(STEPS):
    for s in sheep:
        neighbors = [n for n in sheep if n != s]
        s.update(neighbors, robot, params, fence)  # 👈 aquí

    mode = robot.compute_action(sheep, goal, params)
    logger.log(t, sheep, robot, mode=mode)

logger.save()