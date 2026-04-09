import numpy as np

class ShepherdRobot:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)

    def compute_action(self, sheep, goal, params):
        positions = np.array([s.position for s in sheep])
        center = np.mean(positions, axis=0)

        # --- medir dispersión ---
        distances = np.linalg.norm(positions - center, axis=1)
        max_dist = np.max(distances)

        # --- decidir modo ---
        if max_dist > params["collect_threshold"]:
            mode = "collect"
        else:
            mode = "drive"

        # --- calcular target ---
        if mode == "collect":
            # oveja más alejada
            target_sheep = sheep[np.argmax(distances)]
            target_pos = target_sheep.position

            # punto detrás de la oveja respecto al centro
            direction = target_pos - center
            direction = direction / (np.linalg.norm(direction) + 1e-6)

            desired_pos = target_pos + direction * params["collect_distance"]

        else:  # driving
            # empujar el rebaño hacia el goal
            direction = center - goal
            direction = direction / (np.linalg.norm(direction) + 1e-6)

            desired_pos = center + direction * params["drive_distance"]

        # --- movimiento del robot ---
        move = desired_pos - self.position
        speed = np.linalg.norm(move)

        if speed > params["robot_max_speed"]:
            move = (move / speed) * params["robot_max_speed"]

        move = desired_pos - self.position
        dist = np.linalg.norm(move)

        if dist > 0:
            direction = move / dist
        else:
            direction = np.zeros(2)

        # velocidad limitada
        step = direction * params["robot_max_speed"] * params["dt"]

        self.position += step

        return mode  # útil para logging/debug