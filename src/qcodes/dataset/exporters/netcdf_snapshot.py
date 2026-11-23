"""Storage of the dataset snapshot in a group of exported NetCDF files.

By default the snapshot is stored as the JSON string attribute ``snapshot`` on
the root group of the NetCDF file. HDF5 attributes cannot be compressed and are
always read when the file is opened, which is costly for large snapshots.

When ``qcodes.config.dataset.export_snapshot_in_group`` is enabled the snapshot
is instead stored as a ``uint8`` variable containing the UTF-8 encoded JSON in
the ``qcodes_metadata`` group, compressed with the same deflate level as the
data (``qcodes.config.dataset.export_netcdf_compression_level``). The root
``snapshot`` attribute is then replaced by a small JSON pointer to that
variable such that older versions of QCoDeS can still load the file and point
users to the location of the snapshot.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

SNAPSHOT_GROUP = "qcodes_metadata"
SNAPSHOT_VARIABLE = "snapshot"
SNAPSHOT_LOCATION = f"{SNAPSHOT_GROUP}/{SNAPSHOT_VARIABLE}"
_SNAPSHOT_DIMENSION = "snapshot_bytes"
_POINTER_KEY = "__qcodes_snapshot_location__"
_SNAPSHOT_POINTER = json.dumps({_POINTER_KEY: SNAPSHOT_LOCATION})


def _snapshot_pointer_location(snapshot_attr: object) -> str | None:
    """Return the location of the snapshot if the attribute is a pointer to it."""
    # The size guard avoids parsing a full JSON snapshot just to detect a pointer.
    if (
        not isinstance(snapshot_attr, str)
        or len(snapshot_attr) > 1000
        or not snapshot_attr.startswith(f'{{"{_POINTER_KEY}"')
    ):
        return None
    location = json.loads(snapshot_attr).get(_POINTER_KEY)
    return location if isinstance(location, str) else None


def _should_write_snapshot_to_group(snapshot_attr: object) -> TypeGuard[str]:
    return (
        isinstance(snapshot_attr, str)
        and snapshot_attr not in ("", "null")
        and _snapshot_pointer_location(snapshot_attr) is None
    )


def _write_snapshot_to_group(
    file_path: str | Path,
    snapshot_raw: str,
    *,
    compression_level: int,
    chunk_size_bytes: int,
) -> None:
    """
    Append the snapshot as a variable to the ``qcodes_metadata`` group of an
    existing NetCDF file, using deflate compression unless ``compression_level``
    is 0.
    """
    import h5netcdf  # type: ignore[import-untyped]

    data = np.frombuffer(snapshot_raw.encode("utf-8"), dtype=np.uint8)
    compression: dict[str, Any] = (
        {
            "chunks": (min(data.size, chunk_size_bytes),),
            "compression": "gzip",
            "compression_opts": compression_level,
        }
        if compression_level > 0
        else {}
    )
    with h5netcdf.File(file_path, mode="r+") as nc_file:
        group = nc_file.create_group(SNAPSHOT_GROUP)
        group.dimensions[_SNAPSHOT_DIMENSION] = data.size
        variable = group.create_variable(
            SNAPSHOT_VARIABLE, (_SNAPSHOT_DIMENSION,), dtype=np.uint8, **compression
        )
        variable[:] = data
        variable.attrs["description"] = (
            "UTF-8 encoded JSON snapshot of the QCoDeS dataset."
        )


def _read_snapshot_from_group(file_path: str | Path, location: str) -> str:
    import h5netcdf  # pyright: ignore[reportMissingTypeStubs]

    with h5netcdf.File(file_path, mode="r") as nc_file:
        variable = nc_file[location]
        if not isinstance(variable, h5netcdf.Variable):
            raise TypeError(
                f"Expected the snapshot at {location!r} in {file_path} to be a variable."
            )
        return variable[...].tobytes().decode("utf-8")


def _resolve_netcdf_snapshot_raw(file_path: str | Path, snapshot_attr: Any) -> Any:
    """Return the raw snapshot, following the pointer to a snapshot group if needed."""
    location = _snapshot_pointer_location(snapshot_attr)
    if location is None:
        return snapshot_attr
    return _read_snapshot_from_group(file_path, location)


def snapshot_from_netcdf(path: str | Path) -> dict[str, Any] | None:
    """
    Read the snapshot of a dataset exported to NetCDF by QCoDeS.

    This works both for files where the snapshot is stored as the ``snapshot``
    attribute and for files where it is stored in the ``qcodes_metadata`` group (see
    ``qcodes.config.dataset.export_snapshot_in_group``). Unlike
    :func:`qcodes.dataset.load_from_netcdf` this does not require access to a
    QCoDeS database.

    Args:
        path: Path to the NetCDF file.

    Returns:
        The snapshot as a dictionary or None if the dataset has no snapshot.

    """
    import h5netcdf  # pyright: ignore[reportMissingTypeStubs]

    with h5netcdf.File(path, mode="r") as nc_file:
        snapshot_attr = nc_file.attrs.get(SNAPSHOT_VARIABLE)
    if snapshot_attr is None:
        return None
    if isinstance(snapshot_attr, bytes):
        snapshot_attr = snapshot_attr.decode("utf-8")
    return json.loads(_resolve_netcdf_snapshot_raw(path, snapshot_attr))
