"""
plan_executor.py
----------------
Maps each symbolic PDDL action onto concrete robot control parameters
and determines when the action has been completed so the plan can advance.

The executor does NOT move the robot directly — it sets the *mode* and
overrides *params* that the ShepherdRobot.compute_action() already
understands.
"""

import numpy as np


class PlanExecutor:
    """
    Tracks the current PDDL action and exposes a method that translates it
    into a robot behaviour mode plus any parameter overrides.
    """

    def __init__(self, abstraction, goal_reached_radius=2.0):
        """
        Parameters
        ----------
        abstraction : WorldAbstraction
            Used to convert positions to zone names during completion checks.
        goal_reached_radius : float
            Euclidean distance threshold at which the flock centroid is
            considered to have reached the goal (for pen-flock completion).
        """
        self.abstraction = abstraction
        self.goal_reached_radius = goal_reached_radius

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _zone_center_from_name(self, zone_name):
        """Convert a zone name like z_1_2 to its continuous center point."""
        try:
            _, row_str, col_str = zone_name.split("_")
            row = int(row_str)
            col = int(col_str)
        except (ValueError, AttributeError):
            return None

        x = self.abstraction.xmin + (col + 0.5) * self.abstraction.cell_w
        y = self.abstraction.ymin + (row + 0.5) * self.abstraction.cell_h
        return np.array([x, y], dtype=float)

    def get_mode(self, action):
        """
        Return the reactive mode string that matches this PDDL action.

        The returned mode is consumed by ShepherdRobot.compute_action()
        (which currently accepts "collect" or "drive") and by the Logger.

        Parameters
        ----------
        action : dict | None
            Current PDDL action dict {"name": ..., "args": [...]}, or None
            if the plan is exhausted.

        Returns
        -------
        str  : "collect", "drive", or "idle"
        """
        if action is None:
            return "idle"

        name = action["name"]

        if name == "move-robot":
            # Robot repositioning — use collect mode so the robot moves
            # purposefully without pushing the flock.
            return "collect"

        if name == "collect-outlier":
            return "collect"

        if name in ("drive-flock", "pen-flock"):
            return "drive"

        return "idle"

    def get_param_overrides(self, action, sheep, robot, goal_pos, params):
        """
        Produce a copy of *params* with any overrides specific to the
        current PDDL action.

        For collect-outlier the robot should aim at the furthest outlier
        rather than the full flock centroid.  We communicate this by
        temporarily lowering collect_threshold so that
        ShepherdRobot.compute_action() always stays in collect mode.

        Returns
        -------
        dict  — modified params copy (original is never mutated)
        """
        overrides = dict(params)
        overrides.pop("forced_target_pos", None)
        overrides.pop("forced_collect_sheep_id", None)
        overrides.pop("forced_drive_goal_pos", None)

        if action is None:
            return overrides

        name = action["name"]

        if name == "collect-outlier":
            # Force collect mode by setting threshold very high
            overrides["collect_threshold"] = 1e9
            if action.get("args"):
                sheep_token = action["args"][0]
                if isinstance(sheep_token, str) and sheep_token.startswith("sheep"):
                    try:
                        overrides["forced_collect_sheep_id"] = int(sheep_token.replace("sheep", ""))
                    except ValueError:
                        pass

        elif name in ("drive-flock", "pen-flock"):
            # Force drive mode by setting threshold to zero
            overrides["collect_threshold"] = 0.0
            if name == "drive-flock" and len(action.get("args", [])) >= 3:
                next_zone = action["args"][2]
                zone_center = self._zone_center_from_name(next_zone)
                if zone_center is not None:
                    overrides["forced_drive_goal_pos"] = zone_center
                # Help the robot reach the behind-flock pushing position faster.
                overrides["robot_max_speed"] = params.get("robot_max_speed", 0.3) * 1.25

        elif name == "move-robot":
            # Force robot to move toward the requested symbolic destination zone.
            overrides["collect_threshold"] = 1e9
            if len(action.get("args", [])) >= 2:
                target_zone = action["args"][1]
                zone_center = self._zone_center_from_name(target_zone)
                if zone_center is not None:
                    overrides["forced_target_pos"] = zone_center

        return overrides

    def action_completed(self, action, sheep, robot, goal_pos):
        """
        Return True when the current PDDL action's postcondition is
        satisfied in the continuous simulation.

        Parameters
        ----------
        action : dict | None
        sheep  : list[Sheep]
        robot  : ShepherdRobot
        goal_pos : np.ndarray

        Returns
        -------
        bool
        """
        if action is None:
            return False

        name = action["name"]
        positions = np.array([s.position for s in sheep])
        centroid = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)

        if name == "move-robot":
            # Completed when robot has reached the target zone centroid
            if len(action["args"]) < 2:
                return True
            target_zone = action["args"][1]
            robot_zone = self.abstraction.zone_name(
                *self.abstraction.pos_to_zone(robot.position)
            )
            return robot_zone == target_zone

        if name == "collect-outlier":
            threshold = self.abstraction.collect_threshold
            if action.get("args"):
                sheep_token = action["args"][0]
                if isinstance(sheep_token, str) and sheep_token.startswith("sheep"):
                    try:
                        target_id = int(sheep_token.replace("sheep", ""))
                        for idx, sheep_obj in enumerate(sheep):
                            if sheep_obj.id == target_id:
                                return bool(distances[idx] <= threshold)
                    except ValueError:
                        pass
            # Fallback when the action does not name a valid sheep id.
            return bool(np.max(distances) <= threshold)

        if name == "drive-flock":
            # Completed when flock centroid has entered the target zone
            if len(action["args"]) < 3:
                return False
            target_zone = action["args"][2]  # ?next-fz
            flock_zone = self.abstraction.zone_name(
                *self.abstraction.pos_to_zone(centroid)
            )
            return flock_zone == target_zone

        if name == "pen-flock":
            # Completed when centroid is within goal_reached_radius of goal
            dist_to_goal = np.linalg.norm(centroid - goal_pos)
            return dist_to_goal <= self.goal_reached_radius

        return False
