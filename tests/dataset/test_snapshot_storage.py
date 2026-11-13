from __future__ import annotations

import json
import math
import os
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import uncertainties  # type: ignore[import-untyped]

import qcodes as qc
from qcodes.dataset import load_by_id, new_data_set
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.data_set_in_memory import DataSetInMem, load_from_netcdf
from qcodes.dataset.database_extract_runs import extract_runs_into_db
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.guids import generate_guid
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.sqlite.connection import atomic
from qcodes.dataset.sqlite.database import connect
from qcodes.dataset.sqlite.db_upgrades.version import get_user_version, set_user_version
from qcodes.dataset.sqlite.queries import (
    add_data_to_dynamic_columns,
    create_run,
    get_raw_run_attributes,
)
from qcodes.dataset.sqlite.snapshot_storage import (
    decode_snapshot,
    encode_snapshot,
    zstd,
)
from qcodes.parameters import ParamSpecBase
from qcodes.utils import NumpyJSONEncoder

if TYPE_CHECKING:
    from qcodes.dataset.experiment_container import Experiment

PREFIX = b"QCoDeS.snapshot\x00\x01"


def stored_snapshot(dataset: DataSet) -> str | bytes | None:
    return dataset.conn.execute(
        "SELECT snapshot FROM runs WHERE run_id = ?", (dataset.run_id,)
    ).fetchone()[0]


@pytest.mark.parametrize("compression", ["none", "zstd"])
@pytest.mark.parametrize(
    "raw",
    [
        ' { "full": [1, 2], "unicode": "\\u00e6" }\n',
        '{"text": "\u00e6"}',
        "null",
        "",
        "{",
    ],
)
def test_codec_preserves_raw_strings(raw: str, compression: str) -> None:
    stored = encode_snapshot(raw, compression)
    assert decode_snapshot(stored) == raw
    assert isinstance(stored, bytes if compression == "zstd" else str)
    assert decode_snapshot(None) is None


@pytest.mark.parametrize("compression", ["none", "zstd"])
def test_write_read_and_reopen(dataset: DataSet, compression: str) -> None:
    qc.config.dataset.snapshot_compression = compression
    assert dataset.snapshot is None
    assert dataset.snapshot_raw is None
    assert dataset.get_metadata("snapshot") is None
    raw = ' { "instruments": {"any_name": {"values": [1, 2]}} }\n'
    dataset.add_snapshot(raw)
    stored = stored_snapshot(dataset)
    if compression == "zstd":
        assert isinstance(stored, bytes)
        assert stored.startswith(PREFIX)
    else:
        assert stored == raw

    qc.config.dataset.snapshot_compression = "none" if compression == "zstd" else "zstd"
    assert dataset.path_to_db is not None
    with closing(connect(dataset.path_to_db)) as conn:
        loaded = load_by_id(dataset.run_id, conn=conn)
        assert isinstance(loaded, DataSet)
        assert loaded.snapshot == json.loads(raw)
        assert loaded.snapshot_raw == loaded._snapshot_raw == raw
        assert loaded.get_metadata("snapshot") == raw
        attrs = get_raw_run_attributes(conn, dataset.guid)
        assert attrs is not None
        assert attrs["snapshot"] == raw
        assert attrs["metadata"] == {}
        assert loaded.metadata == {}


def test_overwrite_null_and_history(
    dataset: DataSet, caplog: pytest.LogCaptureFixture
) -> None:
    dataset.add_snapshot("null")
    assert dataset.snapshot is None
    assert dataset.snapshot_raw == "null"
    dataset.add_snapshot('{"nested": {"values": [1, 2]}}')
    historical = new_data_set("historical")
    historical.add_snapshot(dataset.snapshot_raw or "")
    dataset.add_snapshot('{"ignored": true}')
    assert "already has a snapshot" in caplog.text
    assert dataset.snapshot == {"nested": {"values": [1, 2]}}
    decoded = dataset.snapshot
    assert decoded is not None
    decoded["nested"]["values"].append(3)
    assert dataset.snapshot == historical.snapshot == {"nested": {"values": [1, 2]}}
    qc.config.dataset.snapshot_compression = "none"
    dataset.add_snapshot('{"new": true}', overwrite=True)
    assert stored_snapshot(dataset) == '{"new": true}'
    assert historical.snapshot == {"nested": {"values": [1, 2]}}


@pytest.mark.parametrize("raw", ["", "{"])
def test_malformed_json_semantics(dataset: DataSet, raw: str) -> None:
    dataset.add_snapshot(raw)
    assert dataset.snapshot_raw == dataset.get_metadata("snapshot") == raw
    with pytest.raises(json.JSONDecodeError):
        _ = dataset.snapshot
    with pytest.raises(json.JSONDecodeError):
        dataset.add_snapshot("{}", overwrite=True)


@pytest.mark.parametrize(
    "stored",
    [
        b"unknown",
        b"QCoDeS.snapshot\x00\x02" + zstd.compress(b"{}"),
        PREFIX,
        PREFIX + zstd.compress(b"{}")[:-1],
        PREFIX + zstd.compress(b"{}") + b"extra",
        PREFIX + zstd.compress(b"{}") + zstd.compress(b"{}"),
        PREFIX + zstd.compress(b"\xff"),
        PREFIX + b"not a zstandard frame",
    ],
)
def test_corrupt_storage_propagates(dataset: DataSet, stored: bytes) -> None:
    dataset.conn.execute(
        "UPDATE runs SET snapshot = ? WHERE run_id = ?", (stored, dataset.run_id)
    )
    dataset.conn.commit()
    errors = (ValueError, zstd.ZstdError, UnicodeDecodeError)
    with pytest.raises(errors):
        _ = dataset.snapshot
    with pytest.raises(errors):
        _ = dataset.snapshot_raw
    with pytest.raises(errors):
        dataset.get_metadata("snapshot")
    with pytest.raises(errors):
        get_raw_run_attributes(dataset.conn, dataset.guid)
    with pytest.raises(errors):
        dataset.add_snapshot("{}", overwrite=True)


def test_checksum_failure() -> None:
    stored = encode_snapshot('{"data": [1, 2, 3]}', "zstd")
    assert isinstance(stored, bytes)
    damaged = stored[:-1] + bytes([stored[-1] ^ 1])
    with pytest.raises(zstd.ZstdError):
        decode_snapshot(damaged)


def test_invalid_policy_rolls_back_create_run(experiment: Experiment) -> None:
    qc.config.dataset.snapshot_compression = "unknown"
    count = experiment.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    with pytest.raises(RuntimeError, match="Rolling back") as exc:
        create_run(
            experiment.conn,
            experiment.exp_id,
            "invalid",
            generate_guid(),
            snapshot_raw="{}",
        )
    assert isinstance(exc.value.__cause__, ValueError)
    assert experiment.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == count
    assert (
        experiment.conn.execute(
            "SELECT run_counter FROM experiments WHERE exp_id = ?", (experiment.exp_id,)
        ).fetchone()[0]
        == 0
    )


@pytest.mark.parametrize("raw", [None, "null", "", '{"complete": [1, 2]}'])
def test_create_run(experiment: Experiment, raw: str | None) -> None:
    _, run_id, _ = create_run(
        experiment.conn,
        experiment.exp_id,
        "created",
        generate_guid(),
        snapshot_raw=raw,
    )
    dataset = DataSet(run_id=run_id, conn=experiment.conn)
    assert dataset.snapshot_raw == raw


def test_legacy_upgrade_and_mixed_storage(dataset: DataSet) -> None:
    assert dataset.path_to_db is not None
    raw = '{"legacy": true}'
    qc.config.dataset.snapshot_compression = "none"
    dataset.add_snapshot(raw)
    set_user_version(dataset.conn, 9)
    with closing(connect(dataset.path_to_db)) as conn:
        assert get_user_version(conn) == 10
        legacy = DataSet(run_id=dataset.run_id, conn=conn)
        assert stored_snapshot(legacy) == raw
        assert legacy.snapshot == {"legacy": True}
        qc.config.dataset.snapshot_compression = "zstd"
        compressed = new_data_set("compressed", conn=conn)
        compressed.add_snapshot('{"compressed": true}')
        assert isinstance(stored_snapshot(compressed), bytes)
        assert stored_snapshot(legacy) == raw
        assert compressed.snapshot == {"compressed": True}
    with closing(connect(dataset.path_to_db)) as conn:
        assert DataSet(run_id=dataset.run_id, conn=conn).snapshot_raw == raw
        assert DataSet(run_id=compressed.run_id, conn=conn).snapshot == {
            "compressed": True
        }


def test_compressed_writes_require_version_10(dataset: DataSet) -> None:
    set_user_version(dataset.conn, 9)
    with pytest.raises(RuntimeError, match="Rolling back") as exc:
        dataset.add_snapshot("{}")
    assert "schema version 10" in str(exc.value.__cause__)
    assert stored_snapshot(dataset) is None


class CustomEncoded:
    def _JSONEncoder(self) -> dict[str, list[int]]:
        return {"custom": [42]}


def test_prepare_preserves_encoder_types(dataset: DataSet) -> None:
    snapshot = {
        "array": np.array([1.0, 2.0]),
        "int": np.int64(2),
        "bool": np.bool_(True),
        "complex": 1 + 2j,
        "uncertain": uncertainties.ufloat(1, 0.1),
        "custom": CustomEncoded(),
        "nan": float("nan"),
        "inf": float("inf"),
        "minus_inf": -float("inf"),
        "negative_zero": -0.0,
        "large_integer": 2**100,
    }
    dataset.prepare(
        snapshot=snapshot,
        interdeps=InterDependencies_(standalones=(ParamSpecBase("p", "numeric"),)),
    )
    expected_raw = json.dumps(snapshot, cls=NumpyJSONEncoder)
    assert dataset.snapshot_raw == expected_raw
    decoded = dataset.snapshot
    assert decoded is not None
    assert math.isnan(decoded.pop("nan"))
    expected = json.loads(expected_raw)
    expected.pop("nan")
    assert decoded == expected
    assert math.copysign(1, decoded["negative_zero"]) == -1
    dataset.mark_completed()


@pytest.mark.parametrize(
    "source_policy,target_policy", [("zstd", "none"), ("none", "zstd")]
)
def test_extract_reencodes_complete_snapshot(
    dataset: DataSet, tmp_path: Path, source_policy: str, target_policy: str
) -> None:
    qc.config.dataset.snapshot_compression = source_policy
    raw = ' {"full": [1, 2], "nested": {"name": "anything"}}\n'
    dataset.add_snapshot(raw)
    dataset.mark_started()
    dataset.mark_completed()
    qc.config.dataset.snapshot_compression = target_policy
    target = tmp_path / "copy.db"
    assert dataset.path_to_db is not None
    extract_runs_into_db(dataset.path_to_db, target, dataset.run_id)
    with closing(connect(target)) as conn:
        copied = DataSet(run_id=1, conn=conn)
        assert copied.snapshot_raw == raw
        assert copied.snapshot == dataset.snapshot
        assert copied.metadata == {}
        assert isinstance(
            stored_snapshot(copied), bytes if target_policy == "zstd" else str
        )


def test_netcdf_export_import_and_metadata_db(
    scalar_dataset: DataSet, tmp_path: Path
) -> None:
    raw = ' {"full": {"values": [1, 2, 3]}}\n'
    scalar_dataset.add_snapshot(raw)
    assert scalar_dataset.to_xarray_dataset().attrs["snapshot"] == raw
    scalar_dataset.export("netcdf", path=tmp_path)
    imported = load_from_netcdf(scalar_dataset.export_info.export_paths["nc"])
    assert imported._snapshot_raw == raw
    assert imported.snapshot == scalar_dataset.snapshot
    assert "snapshot" not in imported.metadata
    target = tmp_path / "imported.db"
    imported.write_metadata_to_db(path_to_db=target)
    with closing(connect(target)) as conn:
        attrs = get_raw_run_attributes(conn, imported.guid)
        assert attrs is not None
        assert attrs["snapshot"] == raw
        assert "snapshot" not in attrs["metadata"]
        loaded = DataSetInMem._load_from_db(conn, imported.guid)
        assert loaded._snapshot_raw == raw
        assert loaded.snapshot == scalar_dataset.snapshot
        loaded.add_snapshot('{"updated": true}', overwrite=True)
        attrs = get_raw_run_attributes(conn, imported.guid)
        assert attrs is not None
        assert attrs["snapshot"] == '{"updated": true}'
        stored = conn.execute("SELECT snapshot FROM runs").fetchone()[0]
        assert isinstance(stored, bytes)


def test_metadata_snapshot_write(dataset: DataSet) -> None:
    raw = '{"legacy_import": true}'
    with atomic(dataset.conn):
        add_data_to_dynamic_columns(dataset.conn, dataset.run_id, {"snapshot": raw})
    assert isinstance(stored_snapshot(dataset), bytes)
    assert dataset.get_metadata("snapshot") == raw


def test_in_memory_snapshot_overwrites(
    experiment: Experiment, caplog: pytest.LogCaptureFixture
) -> None:
    dataset = DataSetInMem._create_new_run(name="in-memory")
    assert dataset.snapshot is None
    assert dataset._snapshot_raw is None
    dataset.add_snapshot("null")
    assert dataset.snapshot is None
    assert dataset._snapshot_raw == "null"
    raw = '{"full": {"values": [1, 2]}}'
    dataset.add_snapshot(raw)
    dataset.add_snapshot('{"ignored": true}')
    assert "already has a snapshot" in caplog.text
    assert dataset._snapshot_raw == raw
    decoded = dataset.snapshot
    assert decoded is not None
    decoded["full"]["values"].append(3)
    assert dataset.snapshot == {"full": {"values": [1, 2]}}
    dataset.add_snapshot('{"updated": true}', overwrite=True)
    attrs = get_raw_run_attributes(experiment.conn, dataset.guid)
    assert attrs is not None
    assert dataset._snapshot_raw == attrs["snapshot"] == '{"updated": true}'
    assert isinstance(
        experiment.conn.execute(
            "SELECT snapshot FROM runs WHERE guid = ?", (dataset.guid,)
        ).fetchone()[0],
        bytes,
    )


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("backend", ["sqlite_main_db", "sqlite_per_dataset_db"])
def test_measurement_writers_and_backends(
    experiment: Experiment, background: bool, backend: str
) -> None:
    qc.config.dataset.raw_data_backend = backend
    measurement = Measurement(exp=experiment)
    measurement.register_custom_parameter("p")
    with measurement.run(write_in_background=background) as saver:
        saver.add_result(("p", 1))
    dataset = saver.dataset
    assert isinstance(dataset, DataSet)
    assert isinstance(stored_snapshot(dataset), bytes)
    assert dataset.snapshot is not None
    assert dataset.get_parameter_data()["p"]["p"].tolist() == [1]
    assert dataset.completed


@pytest.mark.skipif(
    not os.environ.get("QCODES_SNAPSHOT_CAPTURE"),
    reason="Opt-in local capture validation; never bundle instrument captures",
)
def test_local_capture_roundtrip(dataset: DataSet) -> None:
    capture = Path(os.environ["QCODES_SNAPSHOT_CAPTURE"])
    snapshot = json.loads(capture.read_text(encoding="utf-8"))
    dataset.prepare(
        snapshot=snapshot,
        interdeps=InterDependencies_(standalones=(ParamSpecBase("p", "numeric"),)),
    )
    expected = json.dumps(snapshot, cls=NumpyJSONEncoder)
    assert dataset.snapshot_raw == expected
    assert dataset.snapshot == snapshot
    stored = stored_snapshot(dataset)
    assert isinstance(stored, bytes)
    raw_size = len(expected.encode("utf-8"))
    assert len(stored) < raw_size / 2
    assert dataset.path_to_db is not None
    with closing(connect(dataset.path_to_db)) as conn:
        loaded = DataSet(run_id=dataset.run_id, conn=conn)
        assert loaded.snapshot == snapshot
        assert loaded.snapshot_raw == expected
    print(
        f"Capture round-trip: JSON={raw_size:,} bytes, BLOB={len(stored):,} bytes, "
        f"reduction={1 - len(stored) / raw_size:.2%}"
    )
    dataset.mark_completed()
