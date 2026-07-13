from __future__ import annotations

from copy import deepcopy
import subprocess
from unittest.mock import patch

import wing_capture_hardware_identity as hardware_module
from wing_capture_hardware_identity import (
    collect_wing_capture_hardware_identity,
    verify_wing_capture_hardware_identity,
)


def _audio_profile(*, transport="coreaudio_device_type_usb"):
    return {
        "SPAudioDataType": [
            {
                "_name": "coreaudio_device",
                "_items": [
                    {
                        "_name": "WING",
                        "coreaudio_device_manufacturer": "BEHRINGER",
                        "coreaudio_device_transport": transport,
                        "coreaudio_device_input": 48,
                        "coreaudio_device_output": 48,
                        "coreaudio_device_srate": 48000,
                    }
                ],
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
    assert result["hardware_uid"].startswith("usb:1397:050b:")
    assert len(result["binding_sha256"]) == 64


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
