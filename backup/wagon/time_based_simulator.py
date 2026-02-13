# wagon/time_based_simulator.py

class TimeBasedWagonSimulator:
    def __init__(self, wagon_interval_sec):
        self.wagon_interval_sec = wagon_interval_sec
        self.current_wagon = 1

    def update(self, current_time_sec):
        """
        Returns wagon_id based on time simulation.
        """
        wagon_id = int(current_time_sec // self.wagon_interval_sec) + 1
        return f"wagon{wagon_id}"
