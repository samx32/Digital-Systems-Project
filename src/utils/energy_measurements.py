# energy_measurements.py

# Library imports
import time
import csv
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime

import psutil
import pynvml
from pyJoules.energy_meter import EnergyContext
from codecarbon import EmissionsTracker


class EnergyTracker:
    """
    A class to track energy consumption using CodeCarbon, PyJoules, and NVIDIA ML.
    Tracks CPU, GPU, and overall system energy consumption during ML training.
    """

    def __init__(self, output_dir: str = "data/results", experiment_name: str = "experiment"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.codecarbon_tracker = None
        self.gpu_available = False

        try:
            pynvml.nvmlInit()
            self.gpu_available = True
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except:
            print("NVIDIA GPU not available for energy tracking.")


        self.metrics = {}


    def start(self):
        """Start energy tracking"""
        # Record start time
        self.start_time = time.time()

        # Start CodeCarbon tracker
        self.codecarbon_tracker = EmissionsTracker(
            output_dir=str(self.output_dir),
            project_name=self.experiment_name,
            log_level="error"
        )
        self.codecarbon_tracker.start()

        # Record initial CPU state
        self.initial_cpu_percent = psutil.cpu_percent()
        self.initial_memory = psutil.virtual_memory().used / (1024 ** 2)  # Convert bytes to MB

        # Record initial GPU state
        if self.gpu_available:
            self.initial_gpu_power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Convert mW to W
            self.initial_gpu_temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            self.initial_gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).used / (1024 ** 2)  # Convert bytes to MB

    def stop(self):
        """Stop energy tracking and save results"""
        # TODO: Stop CodeCarbon tracker
        # TODO: Calculate total time
        # TODO: Get final GPU metrics (if available)
        # TODO: Save to CSV
        pass
    



    