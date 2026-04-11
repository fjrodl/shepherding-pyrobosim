"""
world_abstraction.py
--------------------
Converts the continuous simulation state (sheep positions, robot position)
into a symbolic PDDL-ready representation based on a discrete grid of zones.

Grid layout example for a 20x20 world with GRID_SIZE=4 (4x4 = 16 zones):
  zone names: "z_0_0" (bottom-left) ... "z_3_3" (top-right)
"""

import numpy as np


class WorldAbstraction:
    """
    Splits the world into a regular GRID_SIZE x GRID_SIZE grid and computes
    PDDL predicates from the continuous simulation state.
    """

    def __init__(self, bounds, grid_size=4, collect_threshold=3.0):
        """
        Parameters
        ----------
        bounds : list[float]
            [xmin, ymin, xmax, ymax]
        grid_size : int
            Number of cells per axis.
        collect_threshold : float
            Max distance of any sheep from the flock centroid before the
            flock is considered dispersed.
        """
        self.xmin, self.ymin, self.xmax, self.ymax = bounds
        self.grid_size = grid_size
        self.collect_threshold = collect_threshold

        self.cell_w = (self.xmax - self.xmin) / grid_size
        self.cell_h = (self.ymax - self.ymin) / grid_size

    # ------------------------------------------------------------------
    # Zone helpers
    # ------------------------------------------------------------------

    def pos_to_zone(self, pos):
        """Return (row, col) grid indices for a continuous position."""
        col = int((pos[0] - self.xmin) / self.cell_w)
        row = int((pos[1] - self.ymin) / self.cell_h)
        col = np.clip(col, 0, self.grid_size - 1)
        row = np.clip(row, 0, self.grid_size - 1)
        return int(row), int(col)

    def zone_name(self, row, col):
        return f"z_{row}_{col}"

    def all_zones(self):
        """Return all zone names."""
        return [
            self.zone_name(r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
        ]

    def adjacent_pairs(self):
        """Return list of (zone_a, zone_b) pairs that are 4-connected neighbors."""
        pairs = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                for dr, dc in [(0, 1), (1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        pairs.append((self.zone_name(r, c), self.zone_name(nr, nc)))
        return pairs

    def goal_zone(self, goal_pos):
        """Return the zone name that contains the goal position."""
        r, c = self.pos_to_zone(goal_pos)
        return self.zone_name(r, c)

    # ------------------------------------------------------------------
    # State abstraction
    # ------------------------------------------------------------------

    def compute(self, sheep, robot, goal_pos):
        """
        Compute the full abstract state from simulation objects.

        Returns
        -------
        dict with keys:
          robot_zone      : str
          flock_zone      : str
          flock_dispersed : bool
          outlier_ids     : list[int]  — sheep ids far from centroid
          sheep_zones     : dict[int, str]  — id → zone name (outliers only)
          goal_zone       : str
          all_zones       : list[str]
          adjacent_pairs  : list[(str,str)]
          behind_triples  : list[(robot_zone, flock_zone, goal_zone)]
        """
        positions = np.array([s.position for s in sheep])
        centroid = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)

        flock_dispersed = bool(np.max(distances) > self.collect_threshold)

        r_rz, c_rz = self.pos_to_zone(robot.position)
        r_fz, c_fz = self.pos_to_zone(centroid)
        r_gz, c_gz = self.pos_to_zone(goal_pos)

        robot_zone = self.zone_name(r_rz, c_rz)
        flock_zone = self.zone_name(r_fz, c_fz)
        goal_zone_ = self.zone_name(r_gz, c_gz)

        # Outlier sheep (distance > threshold)
        outlier_ids = [sheep[i].id for i in range(len(sheep))
                       if distances[i] > self.collect_threshold]
        sheep_zones = {
            sheep[i].id: self.zone_name(*self.pos_to_zone(sheep[i].position))
            for i in range(len(sheep))
            if sheep[i].id in outlier_ids
        }

        # behind_flock triples: robot zone is "behind" flock relative to goal
        # Defined as: robot zone is on the opposite side of the flock from the goal.
        behind_triples = self._compute_behind_triples(goal_zone_)

        return {
            "robot_zone": robot_zone,
            "flock_zone": flock_zone,
            "flock_dispersed": flock_dispersed,
            "outlier_ids": outlier_ids,
            "sheep_zones": sheep_zones,
            "goal_zone": goal_zone_,
            "all_zones": self.all_zones(),
            "adjacent_pairs": self.adjacent_pairs(),
            "behind_triples": behind_triples,
        }

    def _compute_behind_triples(self, goal_zone_name):
        """
        For every (robot_zone, flock_zone) pair, determine if robot_zone is
        'behind' flock_zone relative to goal_zone_name.

        A robot zone is behind the flock when it is on the opposite side of the
        flock from the goal (i.e., the flock is between the robot and the goal).

        Returns list of (robot_zone, flock_zone, goal_zone).
        """
        r_gz, c_gz = map(int, goal_zone_name.split("_")[1:])
        triples = []

        for r_f in range(self.grid_size):
            for c_f in range(self.grid_size):
                flock_zone = self.zone_name(r_f, c_f)
                # Direction from flock toward goal
                dr_fg = r_gz - r_f
                dc_fg = c_gz - c_f

                for r_r in range(self.grid_size):
                    for c_r in range(self.grid_size):
                        robot_zone = self.zone_name(r_r, c_r)
                        # Direction from flock toward robot
                        dr_fr = r_r - r_f
                        dc_fr = c_r - c_f
                        # Dot product: negative means opposite side
                        dot = dr_fg * dr_fr + dc_fg * dc_fr
                        if dot < 0:
                            triples.append((robot_zone, flock_zone, goal_zone_name))

        return triples
