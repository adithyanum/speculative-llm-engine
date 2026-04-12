import json
import os
from datetime import datetime


class MetricsLogger:
    def __init__(self, log_file="metrics/logs.json"):
        self.log_file = log_file

        # create file if not exists
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                json.dump([], f)

    def log(self, data: dict):
        """
        Append new log entry
        """
        with open(self.log_file, "r") as f:
            logs = json.load(f)

        data["timestamp"] = datetime.now().isoformat()

        logs.append(data)

        with open(self.log_file, "w") as f:
            json.dump(logs, f, indent=4)

    def read_logs(self):
        with open(self.log_file, "r") as f:
            return json.load(f)