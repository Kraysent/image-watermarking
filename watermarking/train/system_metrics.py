import logging
import re
import subprocess
import sys

import torch
from mlflow.system_metrics.metrics.base_metrics_monitor import BaseMetricsMonitor
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor

_logger = logging.getLogger(__name__)

_ACTIVE_PATTERN = re.compile(
    r"GPU\s+(?:HW\s+)?Active\s+(?:occupancy|residency):\s+(\d+(?:\.\d+)?)%"
    r"(?:\s*\(([^)]*)\))?",
    re.IGNORECASE,
)
_IDLE_PATTERN = re.compile(r"GPU\s+Idle\s+residency:\s+(\d+(?:\.\d+)?)%", re.IGNORECASE)
_FREQ_ENTRY_PATTERN = re.compile(r"(\d+)\s*MHz:\s*(\d*\.?\d+)\s*%", re.IGNORECASE)
_SW_REQUESTED_PATTERN = re.compile(r"GPU\s+SW\s+requested\s+state:\s*\(([^)]*)\)", re.IGNORECASE)
_SW_STATE_PATTERN = re.compile(r"GPU\s+SW\s+state:\s*\(([^)]*)\)", re.IGNORECASE)
_P_ENTRY_PATTERN = re.compile(r"(?:SW_)?P(\d+)\s*:\s*(\d*\.?\d+)\s*%", re.IGNORECASE)


class MacGPUMonitor(BaseMetricsMonitor):
    def __init__(self) -> None:
        super().__init__()
        self._powermetrics_enabled = True

    def collect_metrics(self) -> None:
        self._collect_mps_memory()
        if not self._powermetrics_enabled:
            return
        metrics = sample_gpu_metrics()
        if metrics is None:
            self._powermetrics_enabled = False
            _logger.info(
                "Mac GPU utilization unavailable (powermetrics needs passwordless sudo); "
                "continuing with MPS memory metrics only."
            )
            return
        for key, value in metrics.items():
            self._metrics[key].append(value)

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


def sample_gpu_metrics() -> dict[str, float] | None:
    commands = (
        ["sudo", "-n", "/usr/bin/powermetrics", "-i", "1000", "-n", "1", "-s", "gpu_power"],
        ["/usr/bin/powermetrics", "-i", "1000", "-n", "1", "-s", "gpu_power"],
    )
    for command in commands:
        output = _run_powermetrics(command)
        if output is None:
            continue
        metrics = parse_gpu_metrics(output)
        if metrics is not None:
            return metrics
    return None


def sample_gpu_utilization() -> float | None:
    metrics = sample_gpu_metrics()
    if metrics is None:
        return None
    return metrics.get("gpu_0_utilization_percentage")


def parse_gpu_utilization(output: str) -> float | None:
    metrics = parse_gpu_metrics(output)
    if metrics is None:
        return None
    return metrics.get("gpu_0_utilization_percentage")


def parse_gpu_metrics(output: str) -> dict[str, float] | None:
    utilization = _parse_utilization(output)
    if utilization is None:
        return None
    metrics: dict[str, float] = {"gpu_0_utilization_percentage": utilization}
    frequency = _parse_weighted_frequency(output)
    if frequency is not None:
        metrics["gpu_0_frequency_mhz"] = frequency
    requested = _parse_dominant_p_state(_SW_REQUESTED_PATTERN.search(output))
    if requested is not None:
        metrics["gpu_0_sw_requested_state"] = float(requested)
    state = _parse_dominant_p_state(_SW_STATE_PATTERN.search(output))
    if state is not None:
        metrics["gpu_0_sw_state"] = float(state)
    return metrics


def _parse_utilization(output: str) -> float | None:
    active_match = _ACTIVE_PATTERN.search(output)
    if active_match is not None:
        return float(active_match.group(1))
    idle_match = _IDLE_PATTERN.search(output)
    if idle_match is not None:
        return max(0.0, 100.0 - float(idle_match.group(1)))
    return None


def _parse_weighted_frequency(output: str) -> float | None:
    active_match = _ACTIVE_PATTERN.search(output)
    if active_match is None or not active_match.group(2):
        return None
    total_weight = 0.0
    weighted_sum = 0.0
    for mhz_text, pct_text in _FREQ_ENTRY_PATTERN.findall(active_match.group(2)):
        weight = float(pct_text)
        if weight <= 0:
            continue
        total_weight += weight
        weighted_sum += float(mhz_text) * weight
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def _parse_dominant_p_state(match: re.Match[str] | None) -> int | None:
    if match is None:
        return None
    best_level: int | None = None
    best_pct = -1.0
    for level_text, pct_text in _P_ENTRY_PATTERN.findall(match.group(1)):
        pct = float(pct_text)
        if pct > best_pct:
            best_pct = pct
            best_level = int(level_text)
    return best_level


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
