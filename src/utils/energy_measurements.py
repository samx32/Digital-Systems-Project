# energy_measurements.py

# Library imports
import time
import csv
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime

import psutil
from py3nvml import py3nvml
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
            py3nvml.nvmlInit()
            self.gpu_available = True
            self.gpu_handle = py3nvml.nvmlDeviceGetHandleByIndex(0)
            print("NVIDIA GPU detected for energy tracking!")
        except Exception as e:
            print(f"NVIDIA GPU not available for energy tracking. Error: {e}")


        self.metrics = {}

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        return False

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
            self.initial_gpu_power = py3nvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Convert mW to W
            self.initial_gpu_temp = py3nvml.nvmlDeviceGetTemperature(self.gpu_handle, py3nvml.NVML_TEMPERATURE_GPU)
            self.initial_gpu_memory = py3nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).used / (1024 ** 2)  # Convert bytes to MB

    def stop(self):
        """Stop energy tracking and save results"""
        # Calculate total time      
        self.end_time = time.time()
        total_time = self.end_time - self.start_time

        # Stop CodeCarbon tracker
        emmissions = self.codecarbon_tracker.stop()

        # Record final CPU state
        final_cpu_percent = psutil.cpu_percent()
        final_memory = psutil.virtual_memory().used / (1024 ** 2)

        # Metrics dictionary
        self.metrics = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "duration_seconds": total_time,
            "emissions_kg": emmissions,
            "initial_cpu_percent": self.initial_cpu_percent,
            "final_cpu_percent": final_cpu_percent,
            "initial_memory_mb": self.initial_memory,
            "final_memory_mb": final_memory,
        }

        # Get final GPU metrics
        if self.gpu_available:
            final_gpu_power = py3nvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
            final_gpu_temp = py3nvml.nvmlDeviceGetTemperature(self.gpu_handle, py3nvml.NVML_TEMPERATURE_GPU)
            final_gpu_memory = py3nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).used / (1024 ** 2)

            self.metrics.update({
                "initial_gpu_power_w": self.initial_gpu_power,
                "final_gpu_power_w": final_gpu_power,
                "initial_gpu_temp_c": self.initial_gpu_temp,
                "final_gpu_temp_c": final_gpu_temp,
                "initial_gpu_memory_mb": self.initial_gpu_memory,
                "final_gpu_memory_mb": final_gpu_memory,
            })

        # Save to CSV
        self._save_to_csv()
        
        return self.metrics
    
    def _save_to_csv(self):
        """Save metrics to a CSV file"""
        csv_file = self.output_dir / f"{self.experiment_name}_energy_metrics_{self.timestamp}.csv"
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.metrics.keys())
            writer.writeheader()
            writer.writerow(self.metrics)

        print(f"Energy metrics saved to {csv_file}")

        
    



    