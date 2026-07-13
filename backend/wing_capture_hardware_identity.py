"""Fail-closed binding between CoreAudio WING capture and OSC console identity."""

from __future__ import annotations

import hashlib
import json
import plistlib
import re
import subprocess
from typing import Any, Mapping, Sequence


WING_USB_VENDOR_ID = 5015
WING_USB_PRODUCT_ID = 1291
WING_CAPTURE_CHANNELS = 48
WING_CAPTURE_SAMPLE_RATE_HZ = 48_000
WING_HARDWARE_IDENTITY_SCHEMA = "automixer.wing_capture_hardware_identity.v1"
MAX_PROFILE_BYTES = 2 * 1024 * 1024


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _blocked(reason: str, **details: Any) -> dict[str, Any]:
    return {
        "schema": WING_HARDWARE_IDENTITY_SCHEMA,
        "verified": False,
        "reason": reason,
        "details": details,
    }


def verify_wing_capture_hardware_identity(
    *,
    device_name: str,
    device_index: int,
    console_serial: str,
    audio_profile: Mapping[str, Any],
    usb_devices: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind one PortAudio stream to one physical WING USB descriptor."""

    if (
        not isinstance(device_name, str)
        or not device_name.strip()
        or isinstance(device_index, bool)
        or not isinstance(device_index, int)
        or device_index < 0
    ):
        return _blocked("capture_device_identity_invalid")
    normalized_console_serial = str(console_serial or "").strip().upper()
    if re.fullmatch(r"[A-Z0-9]{8,32}", normalized_console_serial) is None:
        return _blocked("console_serial_invalid")

    audio_rows: list[Mapping[str, Any]] = []
    profile_rows = audio_profile.get("SPAudioDataType")
    if isinstance(profile_rows, list):
        for profile_row in profile_rows:
            if not isinstance(profile_row, Mapping):
                continue
            items = profile_row.get("_items")
            if isinstance(items, list):
                audio_rows.extend(item for item in items if isinstance(item, Mapping))
            elif profile_row.get("_name"):
                audio_rows.append(profile_row)
    exact_name_rows = [
        row for row in audio_rows if row.get("_name") == device_name.strip()
    ]
    if len(exact_name_rows) != 1:
        return _blocked(
            "coreaudio_capture_name_not_unique",
            exact_name_candidate_count=len(exact_name_rows),
        )
    audio = exact_name_rows[0]
    audio_checks = {
        "manufacturer": str(audio.get("coreaudio_device_manufacturer") or "")
        .strip()
        .upper()
        == "BEHRINGER",
        "usb_transport": audio.get("coreaudio_device_transport")
        == "coreaudio_device_type_usb",
        "input_channels": audio.get("coreaudio_device_input")
        == WING_CAPTURE_CHANNELS,
        "sample_rate": audio.get("coreaudio_device_srate")
        == WING_CAPTURE_SAMPLE_RATE_HZ,
    }
    if not all(audio_checks.values()):
        return _blocked("coreaudio_capture_not_physical_wing", checks=audio_checks)

    wing_named_usb = [
        row
        for row in usb_devices
        if isinstance(row, Mapping)
        and str(
            row.get("USB Product Name")
            or row.get("kUSBProductString")
            or row.get("IORegistryEntryName")
            or ""
        )
        .strip()
        .upper()
        == "WING"
    ]
    if len(wing_named_usb) != 1:
        return _blocked(
            "physical_wing_usb_not_unique",
            wing_usb_candidate_count=len(wing_named_usb),
        )
    usb = wing_named_usb[0]
    usb_checks = {
        "manufacturer": str(
            usb.get("USB Vendor Name") or usb.get("kUSBVendorString") or ""
        )
        .strip()
        .upper()
        == "BEHRINGER",
        "vendor_id": usb.get("idVendor") == WING_USB_VENDOR_ID,
        "product_id": usb.get("idProduct") == WING_USB_PRODUCT_ID,
    }
    if not all(usb_checks.values()):
        return _blocked("physical_wing_usb_descriptor_mismatch", checks=usb_checks)
    usb_serial = str(
        usb.get("USB Serial Number") or usb.get("kUSBSerialNumberString") or ""
    ).strip().upper()
    # This console exposes the OSC/discovery chassis serial and appends one
    # three-digit USB-interface suffix in its USB descriptor.  Both the exact
    # form and that explicit interface form are accepted; arbitrary prefixes
    # are not.
    serial_matches = bool(
        usb_serial == normalized_console_serial
        or re.fullmatch(
            re.escape(normalized_console_serial) + r"[0-9]{3}",
            usb_serial,
        )
    )
    if not serial_matches:
        return _blocked(
            "wing_usb_console_serial_mismatch",
            console_serial=normalized_console_serial,
            usb_serial=usb_serial,
        )
    location_id = usb.get("locationID")
    if (
        isinstance(location_id, bool)
        or not isinstance(location_id, int)
        or location_id < 0
    ):
        return _blocked("wing_usb_location_id_invalid")

    binding = {
        "schema": WING_HARDWARE_IDENTITY_SCHEMA,
        "device_name": device_name.strip(),
        "device_index": device_index,
        "console_serial": normalized_console_serial,
        "coreaudio": {
            "manufacturer": "BEHRINGER",
            "transport": "usb",
            "input_channels": WING_CAPTURE_CHANNELS,
            "sample_rate_hz": WING_CAPTURE_SAMPLE_RATE_HZ,
        },
        "usb": {
            "vendor_id": WING_USB_VENDOR_ID,
            "product_id": WING_USB_PRODUCT_ID,
            "serial": usb_serial,
            "location_id": location_id,
        },
        "serial_binding": (
            "exact"
            if usb_serial == normalized_console_serial
            else "chassis_serial_plus_three_digit_usb_interface"
        ),
    }
    binding_sha256 = hashlib.sha256(_canonical_json_bytes(binding)).hexdigest()
    return {
        **binding,
        "verified": True,
        "reason": "physical_wing_usb_capture_bound_to_console",
        "hardware_uid": (
            f"usb:{WING_USB_VENDOR_ID:04x}:{WING_USB_PRODUCT_ID:04x}:"
            f"{usb_serial}:{location_id:08x}"
        ),
        "binding_sha256": binding_sha256,
    }


def _run_bounded(command: Sequence[str]) -> bytes:
    completed = subprocess.run(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=5,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"hardware_identity_command_failed:{command[0]}")
    if not 0 < len(completed.stdout) <= MAX_PROFILE_BYTES:
        raise RuntimeError("hardware_identity_command_output_size_invalid")
    return completed.stdout


def collect_wing_capture_hardware_identity(
    *,
    device_name: str,
    device_index: int,
    console_serial: str,
) -> dict[str, Any]:
    """Collect macOS-local CoreAudio/USB identity with bounded commands."""

    try:
        audio_profile = json.loads(
            _run_bounded(
                ("/usr/sbin/system_profiler", "SPAudioDataType", "-json")
            ).decode("utf-8")
        )
        usb_devices = plistlib.loads(
            _run_bounded(
                (
                    "/usr/sbin/ioreg",
                    "-p",
                    "IOUSB",
                    "-r",
                    "-c",
                    "IOUSBHostDevice",
                    "-l",
                    "-w",
                    "0",
                    "-a",
                )
            )
        )
        if not isinstance(audio_profile, Mapping) or not isinstance(
            usb_devices,
            list,
        ):
            raise ValueError("hardware identity payload shape invalid")
    except Exception as exc:
        return _blocked(
            "wing_capture_hardware_identity_collection_failed",
            error=type(exc).__name__,
        )
    return verify_wing_capture_hardware_identity(
        device_name=device_name,
        device_index=device_index,
        console_serial=console_serial,
        audio_profile=audio_profile,
        usb_devices=usb_devices,
    )
