import numpy as np


class ShepherdRobot:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)

    @staticmethod
    def _closest_point_on_segment(point, a, b):
        ab = b - a
        denom = np.dot(ab, ab) + 1e-9
        t = np.dot(point - a, ab) / denom
        t = np.clip(t, 0.0, 1.0)
        return a + t * ab

    @staticmethod
    def _segments_intersect(p1, p2, q1, q2):
        def orient(a, b, c):
            return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

        o1 = orient(p1, p2, q1)
        o2 = orient(p1, p2, q2)
        o3 = orient(q1, q2, p1)
        o4 = orient(q1, q2, p2)
        return (o1 != o2) and (o3 != o4)

    def _enforce_sheep_clearance(self, desired_pos, sheep, min_distance):
        adjusted = desired_pos.copy()
        for _ in range(2):
            for s in sheep:
                diff = adjusted - s.position
                dist = np.linalg.norm(diff)
                if dist < min_distance:
                    if dist < 1e-9:
                        diff = adjusted - self.position
                        dist = np.linalg.norm(diff)
                        if dist < 1e-9:
                            diff = np.array([1.0, 0.0])
                            dist = 1.0
                    adjusted = s.position + (diff / dist) * min_distance
        return adjusted

    def _enforce_fence_constraint(self, start_pos, desired_pos, fence, clearance):
        if fence is None:
            return desired_pos

        a = np.array(fence[0], dtype=float)
        b = np.array(fence[1], dtype=float)
        adjusted = desired_pos.copy()

        # If planned segment crosses the fence, slide along fence tangent instead.
        if self._segments_intersect(start_pos, adjusted, a, b):
            fence_vec = b - a
            fence_len = np.linalg.norm(fence_vec)
            if fence_len > 1e-9:
                tangent = fence_vec / fence_len
                motion = adjusted - start_pos
                tangential = np.dot(motion, tangent)
                if abs(tangential) < 1e-9:
                    tangential = np.linalg.norm(motion)
                adjusted = start_pos + tangent * tangential
            else:
                return start_pos.copy()

        # Keep a minimum distance from the fence segment.
        closest = self._closest_point_on_segment(adjusted, a, b)
        diff = adjusted - closest
        dist = np.linalg.norm(diff)
        if dist < clearance:
            if dist < 1e-9:
                fence_vec = b - a
                normal = np.array([-fence_vec[1], fence_vec[0]], dtype=float)
                norm = np.linalg.norm(normal)
                if norm < 1e-9:
                    normal = np.array([0.0, 1.0])
                else:
                    normal = normal / norm
            else:
                normal = diff / dist
            adjusted = closest + normal * clearance

        return adjusted

    def compute_action(self, sheep, goal, params, fence=None):
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

        sheep_clearance = params.get("robot_sheep_min_distance", 1.0)
        fence_clearance = params.get("robot_fence_clearance", 0.35)

        desired_pos = self._enforce_sheep_clearance(desired_pos, sheep, sheep_clearance)
        desired_pos = self._enforce_fence_constraint(self.position, desired_pos, fence, fence_clearance)

        # --- movimiento del robot ---
        move = desired_pos - self.position
        dist = np.linalg.norm(move)
        max_step = params["robot_max_speed"] * params["dt"]

        if dist > 1e-9:
            step = (move / dist) * min(dist, max_step)
            self.position += step

        return mode  # útil para logging/debug