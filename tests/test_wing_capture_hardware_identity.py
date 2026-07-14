from __future__ import annotations

from copy import deepcopy
import hashlib
import subprocess
from unittest.mock import patch

import pytest

import wing_capture_hardware_identity as hardware_module
from wing_capture_hardware_identity import (
    collect_wing_capture_hardware_identity,
    verify_wing_capture_hardware_identity,
)


_MISSING = object()


def _audio_profile(
    *,
    transport="coreaudio_device_type_usb",
    input_channels=48,
    output_channels=48,
    sample_rate=48000,
):
    audio = {
        "_name": "WING",
        "coreaudio_device_manufacturer": "BEHRINGER",
        "coreaudio_device_transport": transport,
        "coreaudio_device_input": input_channels,
        "coreaudio_device_srate": sample_rate,
    }
    if output_channels is not _MISSING:
        audio["coreaudio_device_output"] = output_channels
    return {
        "SPAudioDataType": [
            {
                "_name": "coreaudio_device",
                "_items": [audio],
            }
        ]
    }


def _usb(serial="0100Q1V0604AAE214"):
    return {
        "IORegistryEntryName": "WING",
        "USB Product Name": "WING",
        "USB Vendor Name": "BEHRINGER",
        "USB Serial Number": serial,
        "idVendor": 5015,
        "idProduct": 1291,
        "locationID": 1048576,
    }


def _verify(audio_profile=None, usb_devices=None, console_serial="0100Q1V0604AAE"):
    return verify_wing_capture_hardware_identity(
        device_name="WING",
        device_index=7,
        console_serial=console_serial,
        audio_profile=audio_profile or _audio_profile(),
        usb_devices=usb_devices or [_usb()],
    )


def test_unique_physical_wing_usb_capture_binds_to_console_serial():
    result = _verify()

    assert result["verified"] is True
    assert result["usb"]["serial"] == "0100Q1V0604AAE214"
    assert result["serial_binding"] == (
        "chassis_serial_plus_three_digit_usb_interface"
    )
    assert result["coreaudio"]["input_channels"] == 48
    assert result["coreaudio"]["output_channels"] == 48
    assert result["hardware_uid"].startswith("usb:1397:050b:")
    binding = {
        key: result[key]
        for key in (
            "schema",
            "device_name",
            "device_index",
            "console_serial",
            "coreaudio",
            "usb",
            "serial_binding",
        )
    }
    assert result["binding_sha256"] == hashlib.sha256(
        hardware_module._canonical_json_bytes(binding)
    ).hexdigest()


@pytest.mark.parametrize(
    "output_channels",
    [
        pytest.param(_MISSING, id="missing"),
        pytest.param(0, id="zero"),
        pytest.param(47, id="too_few"),
        pytest.param(49, id="too_many"),
        pytest.param(False, id="false"),
        pytest.param(True, id="true"),
        pytest.param("48", id="string"),
        pytest.param(48.0, id="float"),
    ],
)
def test_non_exact_output_channel_metadata_is_rejected(output_channels):
    result = _verify(
        audio_profile=_audio_profile(output_channels=output_channels)
    )

    assert result["verified"] is False
    assert result["reason"] == "coreaudio_capture_not_physical_wing"
    assert result["details"]["checks"]["output_channels"] is False
    assert "binding_sha256" not in result
    assert "hardware_uid" not in result


@pytest.mark.parametrize("input_channels", ["48", 48.0, True])
def test_non_integer_input_channel_metadata_is_rejected(input_channels):
    result = _verify(
        audio_profile=_audio_profile(input_channels=input_channels)
    )

    assert result["verified"] is False
    assert result["reason"] == "coreaudio_capture_not_physical_wing"
    assert result["details"]["checks"]["input_channels"] is False
    assert "binding_sha256" not in result
    assert "hardware_uid" not in result


@pytest.mark.parametrize("sample_rate", ["48000", 48_000.0, True])
def test_non_integer_sample_rate_metadata_is_rejected(sample_rate):
    result = _verify(audio_profile=_audio_profile(sample_rate=sample_rate))

    assert result["verified"] is False
    assert result["reason"] == "coreaudio_capture_not_physical_wing"
    assert result["details"]["checks"]["sample_rate"] is False
    assert "binding_sha256" not in result
    assert "hardware_uid" not in result


def test_virtual_same_name_coreaudio_device_is_rejected():
    profile = _audio_profile()
    virtual = deepcopy(profile["SPAudioDataType"][0]["_items"][0])
    virtual["coreaudio_device_transport"] = "coreaudio_device_type_virtual"
    profile["SPAudioDataType"][0]["_items"].append(virtual)

    result = _verify(audio_profile=profile)

    assert result["verified"] is False
    assert result["reason"] == "coreaudio_capture_name_not_unique"


def test_second_physical_wing_usb_is_rejected():
    result = _verify(usb_devices=[_usb(), _usb("0100Q1V9999ZZZ214")])

    assert result["verified"] is False
    assert result["reason"] == "physical_wing_usb_not_unique"


def test_wrong_vendor_product_or_serial_is_rejected():
    wrong_id = _usb()
    wrong_id["idProduct"] = 999
    assert _verify(usb_devices=[wrong_id])["reason"] == (
        "physical_wing_usb_descriptor_mismatch"
    )

    wrong_serial = _verify(usb_devices=[_usb("0100Q1V9999ZZZ214")])
    assert wrong_serial["verified"] is False
    assert wrong_serial["reason"] == "wing_usb_console_serial_mismatch"


def test_arbitrary_longer_serial_prefix_is_not_accepted():
    result = _verify(usb_devices=[_usb("0100Q1V0604AAE2149")])

    assert result["verified"] is False
    assert result["reason"] == "wing_usb_console_serial_mismatch"


def test_hardware_collection_timeout_fails_closed():
    with patch.object(
        hardware_module,
        "_run_bounded",
        side_effect=subprocess.TimeoutExpired("system_profiler", 5),
    ):
        result = collect_wing_capture_hardware_identity(
            device_name="WING",
            device_index=7,
            console_serial="0100Q1V0604AAE",
        )

    assert result["verified"] is False
    assert result["reason"] == "wing_capture_hardware_identity_collection_failed"
    assert result["details"]["error"] == "TimeoutExpired"


def test_malformed_system_profile_fails_closed():
    with patch.object(
        hardware_module,
        "_run_bounded",
        return_value=b"not-json",
    ):
        result = collect_wing_capture_hardware_identity(
            device_name="WING",
            device_index=7,
            console_serial="0100Q1V0604AAE",
        )

    assert result["verified"] is False
    assert result["reason"] == "wing_capture_hardware_identity_collection_failed"
    assert result["details"]["error"] == "JSONDecodeError"
