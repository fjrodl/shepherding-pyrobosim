import numpy as np


class Sheep:
    def __init__(self, id, position):
        self.id = id
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(2)

    def avoid_fence(self, fence, threshold=0.8):
        p = self.position
        a = np.array(fence[0])
        b = np.array(fence[1])

        ab = b - a
        ap = p - a

        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-6)
        t = np.clip(t, 0, 1)

        closest = a + t * ab
        dist_vec = p - closest
        dist = np.linalg.norm(dist_vec)

        if dist < threshold:
            # repulsión fuerte + rebote
            force = dist_vec / (dist + 1e-6)
            self.velocity += force * 0.5
            return force
        return np.zeros(2)

    def update(self, neighbors, robot, params, fence=None):
        # --- Vecinos locales (CLAVE para subgrupos) ---
        local_neighbors = [
            n for n in neighbors
            if np.linalg.norm(n.position - self.position) < params["neighbor_radius"]
        ]

        # --- Cohesion ---
        if local_neighbors:
            cohesion = np.mean([n.position for n in local_neighbors], axis=0) - self.position
        else:
            cohesion = np.zeros(2)

        # --- Separation (evitar solapamiento) ---
        separation = np.zeros(2)
        for n in local_neighbors:
            diff = self.position - n.position
            dist = np.linalg.norm(diff)

            if dist < params["min_dist"]:
                separation += diff / (dist + 1e-6)

        # --- Alignment ---
        if local_neighbors:
            alignment = np.mean([n.velocity for n in local_neighbors], axis=0)
        else:
            alignment = np.zeros(2)

        # --- Evitar robot ---
        dist_robot = np.linalg.norm(self.position - robot.position)

        if dist_robot < params["robot_influence"]:
            direction = (self.position - robot.position) / (dist_robot + 1e-6)
            strength = (params["robot_influence"] - dist_robot) / params["robot_influence"]
            avoid_robot = direction * strength
        else:
            avoid_robot = np.zeros(2)

        # --- Evitar valla ---
        fence_force = np.zeros(2)
        if fence is not None:
            fence_force = self.avoid_fence(fence)

        

        # --- Combinación de fuerzas ---
        self.velocity = (
            params["w_coh"] * cohesion +
            params["w_sep"] * separation +
            params["w_align"] * alignment +
            params["w_robot"] * avoid_robot +
            params.get("w_fence", 0.0) * fence_force
        )

        # --- Límite de velocidad (MUY importante) ---
        speed = np.linalg.norm(self.velocity)
        noise = np.random.normal(0, 0.01, 2)
        self.velocity += noise
        if speed > params["max_speed"]:
            self.velocity = (self.velocity / speed) * params["max_speed"]

        # --- Actualizar posición ---
        self.position += self.velocity * params["dt"]

        # límites del mundo
        if "bounds" in params:
            self.enforce_bounds(params["bounds"])

    def enforce_bounds(self, bounds):
        """
        Rebote en los límites del mundo
        bounds = [xmin, ymin, xmax, ymax]
        """
        xmin, ymin, xmax, ymax = bounds

        for i in range(2):
            if self.position[i] < bounds[i]:
                self.position[i] = bounds[i]
                self.velocity[i] *= -0.5  # rebote amortiguado

            if self.position[i] > bounds[i+2]:
                self.position[i] = bounds[i+2]
                self.velocity[i] *= -0.5