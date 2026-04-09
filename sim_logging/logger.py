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
        entry = {
            "time": t,
            "robot": robot.position.tolist(),
            "sheep": [s.position.tolist() for s in sheep],
            "mode": mode  # ahora sí existe
        }
        self.data.append(entry)

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

        print(f"[Logger] Saved log to {self.path}")

    def clear(self):
        self.data = []