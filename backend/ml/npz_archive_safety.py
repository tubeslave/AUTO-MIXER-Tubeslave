"""Pre-allocation resource validation for numeric NPZ training datasets."""

from __future__ import annotations

import math
import io
import os
import stat
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


MAX_NPZ_COMPRESSED_BYTES = 32 * 1024 * 1024
MAX_NPZ_MEMBER_COUNT = 8
MAX_NPZ_ARRAY_BYTES = 192 * 1024 * 1024
MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_NPZ_ARRAY_ELEMENTS = 48_000_000
MAX_NPZ_ARRAY_DIMENSIONS = 4
MAX_NPY_HEADER_BYTES = 10_000
MAX_NPZ_TEXT_ITEM_BYTES = 256
ALLOWED_NPZ_COMPRESSION = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})


@contextmanager
def open_regular_binary_file(
    path: str | os.PathLike[str],
    *,
    error_message: str,
):
    """Open without following links or blocking on a FIFO, then require regular."""

    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(error_message) from exc
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ValueError(error_message)
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            yield source
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_numeric_npy_header(
    stream,
    *,
    member_name: str,
    allow_bounded_text: bool = False,
) -> tuple[tuple[int, ...], np.dtype, int]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(
            stream, max_header_size=MAX_NPY_HEADER_BYTES
        )
    elif version == (2, 0):
        shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(
            stream, max_header_size=MAX_NPY_HEADER_BYTES
        )
    else:
        raise ValueError(f"Unsupported NPY version in {member_name}: {version!r}")

    if not isinstance(shape, tuple) or len(shape) > MAX_NPZ_ARRAY_DIMENSIONS:
        raise ValueError(f"NPZ array has unsafe dimensions: {member_name}")
    if any(type(item) is not int or item < 0 for item in shape):
        raise ValueError(f"NPZ array has unsafe shape: {member_name}")
    if any(item > MAX_NPZ_ARRAY_ELEMENTS for item in shape):
        raise ValueError(f"NPZ array dimension limit exceeded: {member_name}")
    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise ValueError("Object arrays cannot be loaded when allow_pickle=False")
    is_bounded_text = (
        allow_bounded_text
        and dtype.kind in {"S", "U"}
        and 0 < dtype.itemsize <= MAX_NPZ_TEXT_ITEM_BYTES
    )
    if dtype.fields is not None or (
        not np.issubdtype(dtype, np.number) and not is_bounded_text
    ):
        raise ValueError(f"NPZ array must use a plain numeric dtype: {member_name}")

    element_count = math.prod(shape) if shape else 1
    if element_count > MAX_NPZ_ARRAY_ELEMENTS:
        raise ValueError(f"NPZ array element limit exceeded: {member_name}")
    payload_bytes = element_count * dtype.itemsize
    if payload_bytes > MAX_NPZ_ARRAY_BYTES:
        raise ValueError(f"NPZ array byte limit exceeded: {member_name}")
    return shape, dtype, payload_bytes


def _validate_npz_stream(
    archive_file,
    *,
    required_key_groups: Sequence[Sequence[str]],
    bounded_text_keys: Sequence[str],
    archive_size: int,
    max_compressed_bytes: int,
) -> None:
    if archive_size <= 0 or archive_size > max_compressed_bytes:
        raise ValueError("NPZ compressed file size limit exceeded")
    archive_file.seek(0)
    if not zipfile.is_zipfile(archive_file):
        raise ValueError("NPZ dataset is not a valid ZIP archive")
    archive_file.seek(0)

    with zipfile.ZipFile(archive_file, mode="r") as archive:
        members = archive.infolist()
        if not members or len(members) > MAX_NPZ_MEMBER_COUNT:
            raise ValueError("NPZ member count limit exceeded")

        keys: set[str] = set()
        total_uncompressed = 0
        for member in members:
            name = member.filename
            if (
                not isinstance(name, str)
                or not name.endswith(".npy")
                or Path(name).name != name
                or "/" in name
                or "\\" in name
                or "\x00" in name
            ):
                raise ValueError(f"Unsafe NPZ member name: {name!r}")
            key = name[:-4]
            if not key or key in keys:
                raise ValueError(f"Duplicate or empty NPZ array key: {key!r}")
            keys.add(key)
            if member.flag_bits & 0x1:
                raise ValueError(f"Encrypted NPZ members are not supported: {name}")
            if member.compress_type not in ALLOWED_NPZ_COMPRESSION:
                raise ValueError(f"Unsupported NPZ compression method: {name}")
            if member.file_size < 0 or member.compress_size < 0:
                raise ValueError(f"Invalid NPZ member size: {name}")
            if member.file_size > MAX_NPZ_ARRAY_BYTES + MAX_NPY_HEADER_BYTES:
                raise ValueError(f"NPZ member byte limit exceeded: {name}")
            total_uncompressed += member.file_size
            if total_uncompressed > MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES:
                raise ValueError("NPZ total uncompressed byte limit exceeded")

            with archive.open(member, mode="r") as stream:
                _shape, _dtype, payload_bytes = _read_numeric_npy_header(
                    stream,
                    member_name=name,
                    allow_bounded_text=key in bounded_text_keys,
                )
                header_bytes = stream.tell()
            if header_bytes + payload_bytes != member.file_size:
                raise ValueError(f"NPZ array payload size mismatch: {name}")

        for group in required_key_groups:
            if not any(key in keys for key in group):
                rendered = " or ".join(repr(key) for key in group)
                raise ValueError(f"NPZ must include {rendered}")


@contextmanager
def load_validated_numeric_npz(
    path: str | os.PathLike[str],
    *,
    required_key_groups: Sequence[Sequence[str]],
    bounded_text_keys: Sequence[str] = (),
    max_compressed_bytes: int | None = None,
) -> Iterator[np.lib.npyio.NpzFile]:
    """Yield an NPZ loaded only from one bounded immutable byte snapshot."""

    if max_compressed_bytes is None:
        effective_max_bytes = MAX_NPZ_COMPRESSED_BYTES
    elif (
        type(max_compressed_bytes) is not int
        or max_compressed_bytes <= 0
    ):
        raise ValueError("max_compressed_bytes must be a positive integer")
    else:
        effective_max_bytes = min(
            max_compressed_bytes,
            MAX_NPZ_COMPRESSED_BYTES,
        )

    with open_regular_binary_file(
        path,
        error_message="NPZ dataset must be a regular file",
    ) as source_file:
        snapshot = source_file.read(effective_max_bytes + 1)
        if len(snapshot) > effective_max_bytes:
            raise ValueError("NPZ compressed file size limit exceeded")

    archive_file = io.BytesIO(snapshot)
    try:
        _validate_npz_stream(
            archive_file,
            required_key_groups=required_key_groups,
            bounded_text_keys=bounded_text_keys,
            archive_size=len(snapshot),
            max_compressed_bytes=effective_max_bytes,
        )
        archive_file.seek(0)
        with np.load(archive_file, allow_pickle=False) as raw:
            yield raw
    finally:
        archive_file.close()
