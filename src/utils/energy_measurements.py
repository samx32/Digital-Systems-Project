# energy_measurements.py

# Library imports
import time
import csv
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import psutil
from py3nvml import py3nvml


# UK grid carbon intensity in gCO2/kWh.
# Source: UK Government GHG Conversion Factors 2025 (DESNZ/DEFRA)
# https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2025
_DEFAULT_CARBON_INTENSITY = 207.0


class EnergyTracker:
    """
    Tracks energy consumption during ML workloads using continuous background
    sampling of CPU and GPU metrics.

    Measurements:
    - GPU power (watts) is read directly from NVIDIA SMI via py3nvml and
      integrated over time to compute GPU energy in Joules.
    - CPU energy is *estimated* by scaling psutil CPU utilisation against a
      configurable TDP value (cpu_tdp_watts). This is an approximation — real
      power draw depends on workload characteristics, frequency scaling, etc.
    - CO2 emissions (kg) are derived from total energy using a regional grid
      carbon intensity factor (default 207 gCO2/kWh for the UK grid).

    A daemon thread polls metrics at a configurable interval (default 1 s).
    The summary CSV contains aggregate statistics (mean, peak, total energy)
    rather than per-sample data.

    Usage:
        with EnergyTracker(experiment_name="my_run", cpu_tdp_watts=45) as tracker:
            # ... training / inference code ...
        print(tracker.metrics)
    """

    def __init__(
        self,
        output_dir: str = "data/results",
        experiment_name: str = "experiment",
        sampling_interval: float = 1.0,
        cpu_tdp_watts: float = 45.0,
        carbon_intensity_gco2_per_kwh: float = _DEFAULT_CARBON_INTENSITY,
    ):
        """
        Initialise the energy tracker.

        Args:
            output_dir: Directory to save result CSVs.
            experiment_name: Label for this experiment run.
            sampling_interval: Seconds between background samples (default 1.0).
            cpu_tdp_watts: Estimated CPU TDP in watts, used to approximate CPU
                           energy from utilisation percentage. Default 45 W
                           because my machine uses a AMD Ryzen 7 6800H.
            carbon_intensity_gco2_per_kwh: Grid carbon intensity in gCO2 per kWh.
                           Default 207 gCO2/kWh (UK grid, DESNZ/DEFRA 2025).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sampling_interval = sampling_interval
        self.cpu_tdp_watts = cpu_tdp_watts
        self.carbon_intensity = carbon_intensity_gco2_per_kwh

        self.gpu_available = False

        try:
            py3nvml.nvmlInit()
            self.gpu_available = True
            self.gpu_handle = py3nvml.nvmlDeviceGetHandleByIndex(0)
            print("NVIDIA GPU detected for energy tracking!")
        except Exception as e:
            print(f"NVIDIA GPU not available for energy tracking. Error: {e}")

        self.metrics: Dict[str, Any] = {}
        self._samples: list = []
        self._stop_event = threading.Event()
        self._sampling_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #

    def set_accuracy(self, accuracy: float) -> None:
        """
        Optionally record model accuracy so it is saved alongside energy data.

        Args:
            accuracy: Model accuracy as a float (e.g. 0.92 for 92 %).
        """
        self.metrics["accuracy"] = accuracy

    # ------------------------------------------------------------------ #
    # Start / stop
    # ------------------------------------------------------------------ #

    def start(self):
        """Start energy tracking and background sampling thread."""
        # Record start time
        self.start_time = time.time()

        # Prime psutil so the first real reading is not 0 %
        psutil.cpu_percent()

        # Reset sample storage and start sampling thread
        self._samples = []
        self._stop_event.clear()
        self._sampling_thread = threading.Thread(
            target=self._sampling_loop, daemon=True
        )
        self._sampling_thread.start()

    def stop(self):
        """Stop sampling, compute aggregates, and save results."""
        # Signal the sampling thread to stop and wait for it
        self._stop_event.set()
        if self._sampling_thread is not None:
            self._sampling_thread.join(timeout=5)

        # Calculate total duration
        self.end_time = time.time()
        total_time = self.end_time - self.start_time

        # Build metrics from continuous samples
        self.metrics = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "duration_seconds": round(total_time, 4),
            "num_samples": len(self._samples),
        }

        self._compute_aggregates()

        # Save to CSV
        self._save_to_csv()

        return self.metrics

    # ------------------------------------------------------------------ #
    # Background sampling
    # ------------------------------------------------------------------ #

    def _sampling_loop(self):
        """Continuously sample CPU/GPU metrics until stop event is set."""
        while not self._stop_event.is_set():
            sample: Dict[str, Any] = {
                "time": time.time(),
                "cpu_percent": psutil.cpu_percent(),
                "memory_mb": psutil.virtual_memory().used / (1024 ** 2),
            }

            if self.gpu_available:
                try:
                    sample["gpu_power_w"] = (
                        py3nvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
                    )
                    sample["gpu_temp_c"] = py3nvml.nvmlDeviceGetTemperature(
                        self.gpu_handle, py3nvml.NVML_TEMPERATURE_GPU
                    )
                    sample["gpu_memory_mb"] = (
                        py3nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).used
                        / (1024 ** 2)
                    )
                except Exception:
                    pass  # GPU query can occasionally fail; skip sample fields

            self._samples.append(sample)
            self._stop_event.wait(timeout=self.sampling_interval)

    # ------------------------------------------------------------------ #
    # Aggregate computation
    # ------------------------------------------------------------------ #

    def _compute_aggregates(self):
        """Derive summary statistics and energy totals from collected samples."""
        if not self._samples:
            return

        # ---- CPU stats ---- #
        cpu_values = [s["cpu_percent"] for s in self._samples]
        mem_values = [s["memory_mb"] for s in self._samples]

        self.metrics.update({
            "avg_cpu_percent": round(sum(cpu_values) / len(cpu_values), 2),
            "peak_cpu_percent": round(max(cpu_values), 2),
            "avg_memory_mb": round(sum(mem_values) / len(mem_values), 2),
            "peak_memory_mb": round(max(mem_values), 2),
        })

        # ---- CPU energy estimate (trapezoidal integration) ---- #
        cpu_energy_j = 0.0
        for i in range(1, len(self._samples)):
            dt = self._samples[i]["time"] - self._samples[i - 1]["time"]
            avg_util = (
                self._samples[i - 1]["cpu_percent"]
                + self._samples[i]["cpu_percent"]
            ) / 2.0
            cpu_power_w = (avg_util / 100.0) * self.cpu_tdp_watts
            cpu_energy_j += cpu_power_w * dt

        self.metrics["cpu_energy_joules"] = round(cpu_energy_j, 4)

        # ---- GPU stats & energy ---- #
        gpu_power_values = [s["gpu_power_w"] for s in self._samples if "gpu_power_w" in s]

        if gpu_power_values:
            gpu_temp_values = [s["gpu_temp_c"] for s in self._samples if "gpu_temp_c" in s]
            gpu_mem_values = [s["gpu_memory_mb"] for s in self._samples if "gpu_memory_mb" in s]

            self.metrics.update({
                "avg_gpu_power_w": round(sum(gpu_power_values) / len(gpu_power_values), 2),
                "peak_gpu_power_w": round(max(gpu_power_values), 2),
                "avg_gpu_temp_c": round(sum(gpu_temp_values) / len(gpu_temp_values), 2) if gpu_temp_values else None,
                "peak_gpu_temp_c": round(max(gpu_temp_values), 2) if gpu_temp_values else None,
                "avg_gpu_memory_mb": round(sum(gpu_mem_values) / len(gpu_mem_values), 2) if gpu_mem_values else None,
                "peak_gpu_memory_mb": round(max(gpu_mem_values), 2) if gpu_mem_values else None,
            })

            # GPU energy (trapezoidal integration of power readings)
            gpu_samples = [
                (s["time"], s["gpu_power_w"])
                for s in self._samples
                if "gpu_power_w" in s
            ]
            gpu_energy_j = 0.0
            for i in range(1, len(gpu_samples)):
                dt = gpu_samples[i][0] - gpu_samples[i - 1][0]
                avg_power = (gpu_samples[i - 1][1] + gpu_samples[i][1]) / 2.0
                gpu_energy_j += avg_power * dt

            self.metrics["gpu_energy_joules"] = round(gpu_energy_j, 4)
        else:
            self.metrics["gpu_energy_joules"] = 0.0

        # ---- Total energy ---- #
        self.metrics["total_energy_joules"] = round(
            self.metrics["cpu_energy_joules"] + self.metrics["gpu_energy_joules"], 4
        )

        # ---- Estimated CO2 emissions ---- #
        # Convert Joules → kWh (1 kWh = 3,600,000 J), then multiply by
        # grid carbon intensity (gCO2/kWh) and convert grams → kg.
        total_kwh = self.metrics["total_energy_joules"] / 3_600_000
        self.metrics["total_energy_kwh"] = round(total_kwh, 8)
        self.metrics["emissions_gco2"] = round(
            total_kwh * self.carbon_intensity, 4
        )
        self.metrics["emissions_kg"] = round(
            self.metrics["emissions_gco2"] / 1000, 8
        )
        self.metrics["carbon_intensity_gco2_per_kwh"] = self.carbon_intensity

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _save_to_csv(self):
        """Save metrics to a CSV file."""
        csv_file = (
            self.output_dir
            / f"{self.experiment_name}_energy_metrics_{self.timestamp}.csv"
        )
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.metrics.keys())
            writer.writeheader()
            writer.writerow(self.metrics)

        print(f"Energy metrics saved to {csv_file}")
