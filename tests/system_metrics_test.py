from unittest import mock

import watermarking.train.system_metrics as system_metrics_module

_SAMPLE_POWERMETRICS = (
    "GPU HW active residency:  97.50% ("
    "338 MHz: .06% 618 MHz:   0% 796 MHz:   0% 924 MHz:   0% 952 MHz:   0% "
    "1056 MHz:  11% 1062 MHz:   0% 1182 MHz:  84% 1182 MHz:   0% 1312 MHz: 3.2% "
    "1242 MHz:   0% 1380 MHz:   0% 1326 MHz:   0% 1470 MHz:   0% 1578 MHz:   0%)\n"
    "GPU SW requested state: ("
    "P1 :   0% P2 :   0% P3 :   0% P4 :   0% P5 :   0% P6 :   0% P7 :   0% "
    "P8 :   0% P9 :   0% P10 : 100% P11 :   0% P12 :   0% P13 :   0% "
    "P14 :   0% P15 :   0%)\n"
    "GPU SW state: ("
    "SW_P1 : .06% SW_P2 :   0% SW_P3 :   0% SW_P4 :   0% SW_P5 :  11% "
    "SW_P6 :  84% SW_P7 : 3.2% SW_P8 :   0% SW_P9 :   0% SW_P10 :   0% "
    "SW_P11 :   0% SW_P12 :   0% SW_P13 :   0% SW_P14 :   0% SW_P15 :   0%)\n"
)


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


def test_parse_gpu_metrics_from_powermetrics_sample() -> None:
    metrics = system_metrics_module.parse_gpu_metrics(_SAMPLE_POWERMETRICS)
    assert metrics is not None
    assert metrics["gpu_0_utilization_percentage"] == 97.5
    assert metrics["gpu_0_sw_requested_state"] == 10.0
    assert metrics["gpu_0_sw_state"] == 6.0
    assert round(metrics["gpu_0_frequency_mhz"], 1) == 1171.6


def test_mac_gpu_monitor_records_utilization(monkeypatch) -> None:
    monkeypatch.setattr(
        system_metrics_module,
        "sample_gpu_metrics",
        lambda: {
            "gpu_0_utilization_percentage": 33.0,
            "gpu_0_frequency_mhz": 1182.0,
            "gpu_0_sw_requested_state": 10.0,
            "gpu_0_sw_state": 6.0,
        },
    )
    monkeypatch.setattr(system_metrics_module.torch.backends.mps, "is_available", lambda: False)
    monitor = system_metrics_module.MacGPUMonitor()
    monitor.collect_metrics()
    assert monitor.aggregate_metrics() == {
        "gpu_0_utilization_percentage": 33.0,
        "gpu_0_frequency_mhz": 1182.0,
        "gpu_0_sw_requested_state": 10.0,
        "gpu_0_sw_state": 6.0,
    }


def test_mac_gpu_monitor_skips_powermetrics_after_failure(monkeypatch) -> None:
    calls = {"count": 0}

    def sample() -> dict[str, float] | None:
        calls["count"] += 1
        return None

    monkeypatch.setattr(system_metrics_module, "sample_gpu_metrics", sample)
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
