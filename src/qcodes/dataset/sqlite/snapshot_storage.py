"""Versioned, lossless storage of snapshot JSON strings in SQLite."""

from __future__ import annotations

import sys

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

_MAGIC = b"QCoDeS.snapshot\x00"
_VERSION_1 = _MAGIC + b"\x01"


def encode_snapshot(snapshot: str, compression: str) -> str | bytes:
    """Encode raw JSON without parsing or changing its encoder semantics.

    Args:
        snapshot: The original JSON string, including any whitespace.
        compression: ``zstd`` for version 1 storage, or ``none`` for legacy TEXT.

    Returns:
        The original string or a versioned BLOB containing one checksummed frame.

    Raises:
        ValueError: If the write policy is unsupported.

    """
    if compression == "none":
        return snapshot
    if compression != "zstd":
        raise ValueError(f"Unsupported snapshot compression: {compression!r}")
    return _VERSION_1 + zstd.compress(
        snapshot.encode("utf-8"),
        options={
            zstd.CompressionParameter.compression_level: 1,
            zstd.CompressionParameter.checksum_flag: 1,
        },
    )


def decode_snapshot(snapshot: str | bytes | None) -> str | None:
    """Return original JSON from legacy TEXT, versioned BLOB, or SQL NULL.

    Args:
        snapshot: The value stored in the snapshot column.

    Returns:
        The original JSON string, or None for SQL NULL.

    Raises:
        ValueError: If the format is unknown, truncated, or has trailing data.
        zstd.ZstdError: If the compressed frame or its checksum is corrupt.
        UnicodeDecodeError: If the decompressed bytes are not UTF-8.

    """
    if snapshot is None or isinstance(snapshot, str):
        return snapshot
    if not isinstance(snapshot, bytes) or not snapshot.startswith(_VERSION_1):
        raise ValueError("Unsupported snapshot storage format or version")
    decompressor = zstd.ZstdDecompressor()
    raw = decompressor.decompress(snapshot[len(_VERSION_1) :])
    if not decompressor.eof or decompressor.unused_data:
        raise ValueError("Incomplete snapshot frame or trailing data")
    return raw.decode("utf-8")
