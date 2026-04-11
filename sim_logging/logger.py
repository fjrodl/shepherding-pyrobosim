import json
import os
from datetime import datetime

class Logger:
    def __init__(self, path="data/logs/run.json", auto_timestamp=False):
        if auto_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(path)
            path = f"{base}_{timestamp}{ext}"

        self.path = path
        self.data = []

        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def log(self, t, sheep, robot, mode=None):
        """
        Guarda un estado de la simulación.
        """
        if isinstance(robot, list):
            robot_positions = [r.position.tolist() for r in robot]
            entry = {
                "time": t,
                "robots": robot_positions,
                "sheep": [s.position.tolist() for s in sheep],
                "mode": mode,
            }
        else:
            entry = {
                "time": t,
                "robot": robot.position.tolist(),
                "sheep": [s.position.tolist() for s in sheep],
                "mode": mode,
            }
        self.data.append(entry)

    def flush(self):
        """Write data to disk without clearing the buffer (for live updates)."""
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def save(self):
        self.flush()
        print(f"[Logger] Saved log to {self.path}")

    def clear(self):
        self.data = []