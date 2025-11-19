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
        # TODO: Start CodeCarbon tracker
        # TODO: Record start time
        # TODO: Record initial CPU/GPU state
        pass

    def stop(self):
        """Stop energy tracking and save results"""
        # TODO: Stop CodeCarbon tracker
        # TODO: Calculate total time
        # TODO: Get final GPU metrics (if available)
        # TODO: Save to CSV
        pass
    



    