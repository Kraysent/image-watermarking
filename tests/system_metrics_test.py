from unittest import mock

import watermarking.train.system_metrics as system_metrics_module


def test_parse_gpu_utilization_active_residency() -> None:
    output = "GPU HW active residency:  42.5%\nGPU Idle residency:  57.5%\n"
    assert system_metrics_module.parse_gpu_utilization(output) == 42.5


def test_parse_gpu_utilization_occupancy() -> None:
    output = "GPU Active occupancy: 18.0%\n"
    assert system_metrics_module.parse_gpu_utilization(output) == 18.0


def test_parse_gpu_utilization_from_idle_only() -> None:
    output = "GPU Idle residency:  75.0%\n"
    assert system_metrics_module.parse_gpu_utilization(output) == 25.0


def test_parse_gpu_utilization_missing() -> None:
    assert system_metrics_module.parse_gpu_utilization("no gpu stats") is None


def test_mac_gpu_monitor_records_utilization(monkeypatch) -> None:
    monkeypatch.setattr(
        system_metrics_module,
        "sample_gpu_utilization",
        lambda: 33.0,
    )
    monkeypatch.setattr(system_metrics_module.torch.backends.mps, "is_available", lambda: False)
    monitor = system_metrics_module.MacGPUMonitor()
    monitor.collect_metrics()
    assert monitor.aggregate_metrics() == {"gpu_0_utilization_percentage": 33.0}


def test_mac_gpu_monitor_skips_powermetrics_after_failure(monkeypatch) -> None:
    calls = {"count": 0}

    def sample() -> float | None:
        calls["count"] += 1
        return None

    monkeypatch.setattr(system_metrics_module, "sample_gpu_utilization", sample)
    monkeypatch.setattr(system_metrics_module.torch.backends.mps, "is_available", lambda: False)
    monitor = system_metrics_module.MacGPUMonitor()
    monitor.collect_metrics()
    monitor.collect_metrics()
    assert calls["count"] == 1
    assert monitor.aggregate_metrics() == {}


def test_create_system_metrics_monitor_includes_mac_gpu_on_darwin(monkeypatch) -> None:
    monkeypatch.setattr(system_metrics_module.sys, "platform", "darwin")
    with mock.patch.object(
        system_metrics_module.SystemMetricsMonitorWithMacGPU,
        "_initialize_gpu_monitor",
        return_value=None,
    ):
        monitor = system_metrics_module.create_system_metrics_monitor("run-id")
    assert isinstance(monitor, system_metrics_module.SystemMetricsMonitorWithMacGPU)
    assert any(isinstance(item, system_metrics_module.MacGPUMonitor) for item in monitor.monitors)


def test_create_system_metrics_monitor_skips_mac_gpu_off_darwin(monkeypatch) -> None:
    monkeypatch.setattr(system_metrics_module.sys, "platform", "linux")
    with mock.patch.object(
        system_metrics_module.SystemMetricsMonitor,
        "_initialize_gpu_monitor",
        return_value=None,
    ):
        monitor = system_metrics_module.create_system_metrics_monitor("run-id")
    assert type(monitor) is system_metrics_module.SystemMetricsMonitor
    assert not any(isinstance(item, system_metrics_module.MacGPUMonitor) for item in monitor.monitors)
