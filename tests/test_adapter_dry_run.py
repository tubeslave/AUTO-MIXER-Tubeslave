import json

from integrations.mixing_station.adapter import MixingStationAdapter
from integrations.mixing_station.config import MixingStationConfig
from integrations.mixing_station.models import AutomixCorrection


def make_config(tmp_path, *, dry_run=True, transport="websocket"):
    return MixingStationConfig.from_mapping({
        "enabled": True,
        "dry_run": dry_run,
        "transport": transport,
        "console_profile": "wing_rack",
        "log_jsonl": str(tmp_path / "mixing_station.jsonl"),
    })


def test_adapter_dry_run_logs_without_sending(tmp_path):
    adapter = MixingStationAdapter(make_config(tmp_path, dry_run=True))
    correction = AutomixCorrection(
        console_profile="wing_rack",
        mode="offline_visualization",
        channel_index=0,
        channel_name="Lead Vocal",
        parameter="fader",
        value=-5.0,
        value_unit="db",
        reason="test dry-run",
    )

    result = adapter.send_correction(correction)
    rows = [
        json.loads(line)
        for line in (tmp_path / "mixing_station.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert result.success is True
    assert result.dry_run is True
    assert result.sent is False
    assert rows[0]["parameter"] == "fader"
    assert rows[0]["dry_run"] is True
    assert rows[0]["data_path"] == "ch.0.mix.lvl"


def test_adapter_blocks_unsupported_parameter(tmp_path):
    adapter = MixingStationAdapter(make_config(tmp_path, dry_run=True))
    correction = AutomixCorrection(
        console_profile="wing_rack",
        mode="offline_visualization",
        channel_index=0,
        parameter="fx.parameter",
        value=0.5,
        value_unit="normalized",
        reason="unsupported",
    )

    result = adapter.send_correction(correction)

    assert result.success is False
    assert result.blocked is True
    assert "unsupported" in result.error


def test_adapter_blocks_read_only_compressor_model(tmp_path):
    adapter = MixingStationAdapter(make_config(tmp_path, dry_run=True))
    correction = AutomixCorrection(
        console_profile="wing_rack",
        mode="offline_visualization",
        channel_index=0,
        parameter="compressor.model",
        value=0,
        value_unit="enum",
        reason="model writes reset Wing dyn parameters",
    )

    result = adapter.send_correction(correction)

    assert result.success is False
    assert result.blocked is True
    assert "read-only" in result.error


def test_adapter_non_dry_run_rest_refuses_unknown_write_endpoint(tmp_path):
    adapter = MixingStationAdapter(make_config(tmp_path, dry_run=False, transport="rest"))
    correction = AutomixCorrection(
        console_profile="wing_rack",
        mode="offline_visualization",
        channel_index=0,
        parameter="fader",
        value=-5.0,
        value_unit="db",
        reason="test live endpoint",
        dry_run=False,
    )

    result = adapter.send_correction(correction)

    assert result.success is False
    assert result.sent is False
    assert "write endpoint is not configured" in result.error
