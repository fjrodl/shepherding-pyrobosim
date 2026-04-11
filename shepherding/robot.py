import numpy as np


class ShepherdRobot:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)

    @staticmethod
    def _compute_clusters(positions, radius):
        """Group sheep positions into proximity clusters."""
        if len(positions) == 0:
            return []

        clusters = []
        unvisited = set(range(len(positions)))

        while unvisited:
            seed = unvisited.pop()
            cluster = [seed]
            frontier = [seed]

            while frontier:
                current = frontier.pop()
                for candidate in list(unvisited):
                    if np.linalg.norm(positions[current] - positions[candidate]) < radius:
                        unvisited.remove(candidate)
                        cluster.append(candidate)
                        frontier.append(candidate)

            clusters.append(cluster)

        return clusters

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
        """
        Simple approach: if desired position is too close to any sheep,
        push it in the average direction away from nearby sheep.
        Gentle enforcement to avoid trapping robot.
        """
        adjusted = desired_pos.copy()
        
        # Find all sheep within critical distance
        offending_sheep = []
        for s in sheep:
            diff = adjusted - s.position
            dist = np.linalg.norm(diff)
            if dist < min_distance:
                offending_sheep.append((s, dist, diff))
        
        if not offending_sheep:
            return adjusted
        
        # Push away from all offending sheep with gentle force
        avg_push = np.zeros(2)
        for s, dist, diff in offending_sheep:
            if dist < 1e-9:
                # On top of sheep; use direction from robot position
                push_dir = adjusted - self.position
                if np.linalg.norm(push_dir) > 1e-9:
                    push_dir = push_dir / np.linalg.norm(push_dir)
                else:
                    push_dir = np.array([1.0, 0.0])
            else:
                push_dir = diff / dist
            
            # Weight by violation amount (gentler than before)
            violation = (min_distance - dist) / min_distance
            avg_push += push_dir * violation
        
        if np.linalg.norm(avg_push) > 1e-9:
            avg_push = avg_push / np.linalg.norm(avg_push)
            # Smaller push distance to avoid over-correction
            adjusted = adjusted + avg_push * (min_distance * 0.05)
        
        return adjusted

    def _enforce_fence_constraint(self, start_pos, desired_pos, fence, clearance):
        """
        Enforce fence constraints with intelligent deflection.
        
        Strategy:
        1. Check if path crosses fence
        2. If crossing: deflect motion to slide along fence or go around
        3. Maintain minimum clearance distance
        """
        if fence is None:
            return desired_pos

        a = np.array(fence[0], dtype=float)
        b = np.array(fence[1], dtype=float)
        adjusted = desired_pos.copy()
        
        # Check if desired position path crosses fence
        if self._segments_intersect(start_pos, adjusted, a, b):
            # Path would cross fence - try to deflect around it
            fence_vec = b - a
            fence_len = np.linalg.norm(fence_vec)
            
            if fence_len > 1e-9:
                # Compute inward and outward normals
                fence_tangent = fence_vec / fence_len
                normal1 = np.array([-fence_vec[1], fence_vec[0]], dtype=float)
                normal1 = normal1 / np.linalg.norm(normal1)
                normal2 = -normal1
                
                # Try both directions perpendicular to fence
                # Choose the one that doesn't cross fence
                motion = adjusted - start_pos
                motion_len = np.linalg.norm(motion)
                
                if motion_len > 1e-9:
                    motion_dir = motion / motion_len
                    
                    # Try deflecting along fence tangent
                    tangential_speed = np.dot(motion, fence_tangent)
                    if abs(tangential_speed) > 1e-9:
                        # Move along fence direction instead
                        adjusted = start_pos + fence_tangent * tangential_speed
                    else:
                        # No tangential component; try perpendicular
                        # Push perpendicular to fence (outward)
                        dot1 = np.dot(motion_dir, normal1)
                        dot2 = np.dot(motion_dir, normal2)
                        
                        if abs(dot1) > abs(dot2):
                            adjusted = start_pos + normal1 * motion_len
                        else:
                            adjusted = start_pos + normal2 * motion_len
        
        # Enforce clearance: ensure position stays at least clearance away from fence
        closest = self._closest_point_on_segment(adjusted, a, b)
        diff = adjusted - closest
        dist = np.linalg.norm(diff)
        
        if dist < clearance:
            if dist < 1e-9:
                # On fence line - push perpendicular
                fence_vec = b - a
                normal = np.array([-fence_vec[1], fence_vec[0]], dtype=float)
                norm_len = np.linalg.norm(normal)
                if norm_len > 1e-9:
                    normal = normal / norm_len
                else:
                    normal = np.array([0.0, 1.0])
            else:
                normal = diff / dist
            
            adjusted = closest + normal * clearance
        
        return adjusted

    def compute_action(self, sheep, goal, params, fence=None):
        positions = np.array([s.position for s in sheep])
        center = np.mean(positions, axis=0)
        neighbor_radius = params.get("neighbor_radius", 2.5)
        robot_index = int(params.get("robot_index", 0))
        clusters = self._compute_clusters(positions, neighbor_radius)
        cluster_centers = [np.mean(positions[cluster], axis=0) for cluster in clusters]
        largest_cluster_idx = max(range(len(clusters)), key=lambda idx: len(clusters[idx]))
        main_center = cluster_centers[largest_cluster_idx]
        forced_target_pos = params.get("forced_target_pos", None)
        forced_collect_sheep_id = params.get("forced_collect_sheep_id", None)
        forced_drive_goal_pos = params.get("forced_drive_goal_pos", None)

        # --- medir dispersión ---
        distances = np.linalg.norm(positions - center, axis=1)
        max_dist = np.max(distances)

        # --- decidir modo ---
        if len(clusters) > 1 or max_dist > params["collect_threshold"]:
            mode = "collect"
        else:
            mode = "drive"

        # --- calcular target ---
        if forced_target_pos is not None:
            desired_pos = np.array(forced_target_pos, dtype=float)
            mode = "move-robot"

        elif mode == "collect":
            target_sheep = None
            if forced_collect_sheep_id is not None:
                for sheep_obj in sheep:
                    if sheep_obj.id == forced_collect_sheep_id:
                        target_sheep = sheep_obj
                        break

            if target_sheep is not None:
                target_pos = target_sheep.position
                reference_center = center
            elif len(clusters) > 1:
                stray_cluster_candidates = sorted(
                    (idx for idx in range(len(clusters)) if idx != largest_cluster_idx),
                    key=lambda idx: np.linalg.norm(cluster_centers[idx] - main_center),
                    reverse=True,
                )
                assigned_cluster_rank = min(robot_index, len(stray_cluster_candidates) - 1)
                stray_cluster_idx = stray_cluster_candidates[assigned_cluster_rank]
                stray_indices = clusters[stray_cluster_idx]
                stray_positions = positions[stray_indices]
                stray_distances = np.linalg.norm(stray_positions - main_center, axis=1)
                boundary_pos = stray_positions[np.argmax(stray_distances)]
                target_pos = boundary_pos
                reference_center = main_center
            else:
                sorted_sheep_indices = np.argsort(distances)[::-1]
                target_sheep_idx = sorted_sheep_indices[min(robot_index, len(sorted_sheep_indices) - 1)]
                target_pos = sheep[target_sheep_idx].position
                reference_center = center

            direction = target_pos - reference_center
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            desired_pos = target_pos + direction * params["collect_distance"]

        else:  # driving
            # empujar el rebaño hacia el goal
            drive_goal = np.array(forced_drive_goal_pos, dtype=float) if forced_drive_goal_pos is not None else goal
            direction = main_center - drive_goal
            direction = direction / (np.linalg.norm(direction) + 1e-6)

            desired_pos = main_center + direction * params["drive_distance"]

        sheep_clearance = params.get("robot_sheep_min_distance", 1.0)
        fence_clearance = params.get("robot_fence_clearance", 0.35)
        bounds = params.get("bounds", [0, 0, 20, 20])
        xmin, ymin, xmax, ymax = bounds
        margin = 0.2  # Margin from boundaries

        desired_pos = self._enforce_sheep_clearance(desired_pos, sheep, sheep_clearance)
        desired_pos = self._enforce_fence_constraint(self.position, desired_pos, fence, fence_clearance)
        
        # Enforce world boundaries AFTER constraints
        desired_pos[0] = np.clip(desired_pos[0], xmin + margin, xmax - margin)
        desired_pos[1] = np.clip(desired_pos[1], ymin + margin, ymax - margin)

        # --- movimiento del robot ---
        move = desired_pos - self.position
        dist = np.linalg.norm(move)
        max_step = params["robot_max_speed"] * params["dt"]

        if dist > 1e-9:
            step = (move / dist) * min(dist, max_step)
            self.position += step

        return mode  # útil para logging/debug