import logging
import re
import subprocess
import sys

import torch
from mlflow.system_metrics.metrics.base_metrics_monitor import BaseMetricsMonitor
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor

_logger = logging.getLogger(__name__)

_GPU_UTIL_PATTERNS = (
    re.compile(r"GPU\s+(?:HW\s+)?Active\s+(?:occupancy|residency):\s+(\d+(?:\.\d+)?)%", re.IGNORECASE),
    re.compile(r"GPU\s+Idle\s+residency:\s+(\d+(?:\.\d+)?)%", re.IGNORECASE),
)


class MacGPUMonitor(BaseMetricsMonitor):
    def __init__(self) -> None:
        super().__init__()
        self._powermetrics_enabled = True

    def collect_metrics(self) -> None:
        self._collect_mps_memory()
        if not self._powermetrics_enabled:
            return
        utilization = sample_gpu_utilization()
        if utilization is None:
            self._powermetrics_enabled = False
            _logger.info(
                "Mac GPU utilization unavailable (powermetrics needs passwordless sudo); "
                "continuing with MPS memory metrics only."
            )
            return
        self._metrics["gpu_0_utilization_percentage"].append(utilization)

    def aggregate_metrics(self) -> dict[str, float]:
        return {key: round(sum(values) / len(values), 1) for key, values in self._metrics.items()}

    def _collect_mps_memory(self) -> None:
        if not torch.backends.mps.is_available():
            return
        allocated_bytes = torch.mps.current_allocated_memory()
        driver_bytes = torch.mps.driver_allocated_memory()
        recommended_bytes = torch.mps.recommended_max_memory()
        self._metrics["gpu_0_memory_usage_megabytes"].append(allocated_bytes / 1e6)
        self._metrics["gpu_0_driver_memory_megabytes"].append(driver_bytes / 1e6)
        if recommended_bytes > 0:
            self._metrics["gpu_0_memory_usage_percentage"].append(allocated_bytes / recommended_bytes * 100)


class SystemMetricsMonitorWithMacGPU(SystemMetricsMonitor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if sys.platform == "darwin":
            self.monitors.append(MacGPUMonitor())


def create_system_metrics_monitor(run_id: str) -> SystemMetricsMonitor:
    if sys.platform == "darwin":
        return SystemMetricsMonitorWithMacGPU(run_id=run_id)
    return SystemMetricsMonitor(run_id=run_id)


def sample_gpu_utilization() -> float | None:
    commands = (
        ["sudo", "-n", "/usr/bin/powermetrics", "-i", "1000", "-n", "1", "-s", "gpu_power"],
        ["/usr/bin/powermetrics", "-i", "1000", "-n", "1", "-s", "gpu_power"],
    )
    for command in commands:
        output = _run_powermetrics(command)
        if output is None:
            continue
        utilization = parse_gpu_utilization(output)
        if utilization is not None:
            return utilization
    return None


def parse_gpu_utilization(output: str) -> float | None:
    active_match = _GPU_UTIL_PATTERNS[0].search(output)
    if active_match is not None:
        return float(active_match.group(1))
    idle_match = _GPU_UTIL_PATTERNS[1].search(output)
    if idle_match is not None:
        return max(0.0, 100.0 - float(idle_match.group(1)))
    return None


def _run_powermetrics(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout + result.stderr
