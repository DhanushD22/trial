# wagon/wagon_manager.py

from wagon.time_based_simulator import TimeBasedWagonSimulator
from config.settings import WAGON_INTERVAL_SEC


class WagonManager:
    def __init__(self, mode="time"):
        self.mode = mode

        if self.mode == "time":
            self.engine = TimeBasedWagonSimulator(WAGON_INTERVAL_SEC)
        else:
            raise ValueError("Unsupported wagon mode")

    def get_wagon_id(self, current_time_sec):
        return self.engine.update(current_time_sec)
