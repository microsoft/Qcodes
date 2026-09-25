import json

import numpy
import pytest

import qcodes as qc
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.snapshot_utils import (
    diff_start_end_snapshot,
    diff_start_end_snapshot_by_id,
)
from qcodes.instrument_drivers.mock_instruments import DummyInstrument
from qcodes.parameters import ManualParameter, Parameter
from qcodes.station import Station
from qcodes.utils import ParameterDiff, format_parameter_diff


@pytest.fixture  # scope is "function" per default
def dac():
    dac = DummyInstrument("dummy_dac", gates=["ch1", "ch2"])
    yield dac
    dac.close()


@pytest.fixture
def dmm():
    dmm = DummyInstrument("dummy_dmm", gates=["v1", "v2"])
    yield dmm
    dmm.close()


@pytest.mark.parametrize("pass_station", (True, False))
def test_station_snapshot_during_measurement(
    experiment, dac, dmm, pass_station
) -> None:
    station = Station()
    station.add_component(dac)
    station.add_component(dmm, "renamed_dmm")

    snapshot_of_station = station.snapshot()

    if pass_station:
        measurement = Measurement(experiment, station)
    else:
        # in this branch of the `if` we expect that `Measurement` object
        # will be initialized with `Station.default` which is equal to the
        # station object that is instantiated above
        measurement = Measurement(experiment)

    measurement.register_parameter(dac.ch1)
    measurement.register_parameter(dmm.v1, setpoints=[dac.ch1])
    snapshot_of_parameters = {
        parameter.short_name: parameter.snapshot() for parameter in (dac.ch1, dmm.v1)
    }
    snapshot_of_parameters.update(
        {
            parameter.register_name: parameter.snapshot()
            for parameter in (dac.ch1, dmm.v1)
        }
    )
    with measurement.run() as data_saver:
        data_saver.add_result((dac.ch1, 7), (dmm.v1, 5))

    # 1. Test `get_metadata('snapshot')` method
    # this is not part of the DatasetProtocol interface
    # but we test it anyway
    json_snapshot_from_dataset = data_saver.dataset.get_metadata("snapshot")  # type: ignore[attr-defined]
    snapshot_from_dataset = json.loads(json_snapshot_from_dataset)

    expected_snapshot = {
        "station": snapshot_of_station,
        "parameters": snapshot_of_parameters,
    }
    assert expected_snapshot == snapshot_from_dataset

    # 2. Test `snapshot_raw` property
    # this is not part of the DatasetProtocol interface
    # but we test it anyway
    assert json_snapshot_from_dataset == data_saver.dataset.snapshot_raw  # type: ignore[attr-defined]

    # 3. Test `snapshot` property

    assert expected_snapshot == data_saver.dataset.snapshot


def test_snapshot_creation_for_types_not_supported_by_builtin_json(experiment) -> None:
    """
    Test that `Measurement`/`Runner`/`DataSaver` infrastructure
    successfully dumps station snapshots in JSON format in cases when the
    snapshot contains data of types that are not supported by python builtin
    `json` module, for example, numpy scalars.
    """
    p1 = ManualParameter("p_np_int32", initial_value=numpy.int32(5))
    p2 = ManualParameter("p_np_float16", initial_value=numpy.float16(5.0))
    p3 = ManualParameter("p_np_array", initial_value=numpy.meshgrid((1, 2), (3, 4)))
    p4 = ManualParameter("p_np_bool", initial_value=numpy.bool_(False))

    station = Station(p1, p2, p3, p4)

    measurement = Measurement(experiment, station)

    # we need at least 1 parameter to be able to run the measurement
    measurement.register_custom_parameter("dummy")

    with measurement.run() as data_saver:
        # we do this in order to create a snapshot of the station and add it
        # to the database
        pass

    snapshot = data_saver.dataset.snapshot
    assert snapshot is not None

    assert 5 == snapshot["station"]["parameters"]["p_np_int32"]["value"]
    assert 5 == snapshot["station"]["parameters"]["p_np_int32"]["raw_value"]

    assert 5.0 == snapshot["station"]["parameters"]["p_np_float16"]["value"]
    assert 5.0 == snapshot["station"]["parameters"]["p_np_float16"]["raw_value"]

    lst = [[[1, 2], [1, 2]], [[3, 3], [4, 4]]]
    assert lst == snapshot["station"]["parameters"]["p_np_array"]["value"]
    assert lst == snapshot["station"]["parameters"]["p_np_array"]["raw_value"]

    assert False is snapshot["station"]["parameters"]["p_np_bool"]["value"]
    assert False is snapshot["station"]["parameters"]["p_np_bool"]["raw_value"]


def test_station_snapshot_in_measurement_refreshes_only_invalid_caches(
    experiment,
) -> None:
    """
    The station snapshot taken by a ``Measurement`` uses ``update="Only_invalid"``
    so that parameters with an invalid cache are refreshed via a single ``get``,
    while parameters with a valid cache are not gotten.
    """
    invalid_calls = {"n": 0}
    valid_calls = {"n": 0}

    def invalid_getter() -> int:
        invalid_calls["n"] += 1
        return 42

    def valid_getter() -> int:
        valid_calls["n"] += 1
        return 99

    p_invalid = Parameter("p_invalid", get_cmd=invalid_getter, set_cmd=None)
    p_valid = Parameter("p_valid", get_cmd=valid_getter, set_cmd=None)

    # add components without updating their snapshot (which would call ``get``)
    station = Station()
    station.add_component(p_invalid, update_snapshot=False)
    station.add_component(p_valid, update_snapshot=False)

    # make ``p_valid``'s cache valid without triggering a ``get``
    p_valid.set(7)

    assert not p_invalid.cache.valid
    assert p_valid.cache.valid
    assert invalid_calls["n"] == 0
    assert valid_calls["n"] == 0

    measurement = Measurement(experiment, station)
    # we need at least 1 parameter to be able to run the measurement
    measurement.register_custom_parameter("dummy")

    with measurement.run() as data_saver:
        pass

    snapshot = data_saver.dataset.snapshot
    assert snapshot is not None
    params = snapshot["station"]["parameters"]

    # invalid cache -> refreshed via a single get
    assert invalid_calls["n"] == 1
    assert params["p_invalid"]["value"] == 42

    # valid cache -> not gotten, cached value used
    assert valid_calls["n"] == 0
    assert params["p_valid"]["value"] == 7


def test_end_snapshot_taken_by_default(experiment, dac, dmm) -> None:
    station = Station()
    station.add_component(dac)
    station.add_component(dmm)

    dac.ch1(1)

    measurement = Measurement(experiment, station)
    measurement.register_parameter(dac.ch1)
    measurement.register_parameter(dmm.v1, setpoints=[dac.ch1])

    with measurement.run() as data_saver:
        data_saver.add_result((dac.ch1, 7), (dmm.v1, 5))
        dac.ch1(10)

    dataset = data_saver.dataset

    start_snapshot = dataset.snapshot
    end_snapshot = dataset.end_snapshot
    assert start_snapshot is not None
    assert end_snapshot is not None

    assert (
        start_snapshot["station"]["instruments"]["dummy_dac"]["parameters"]["ch1"][
            "value"
        ]
        == 1
    )
    assert (
        end_snapshot["station"]["instruments"]["dummy_dac"]["parameters"]["ch1"][
            "value"
        ]
        == 10
    )

    # the end snapshot is stored as metadata
    assert dataset.metadata["end_snapshot"] == json.dumps(end_snapshot)


def test_end_snapshot_can_be_disabled(experiment, dac, dmm) -> None:
    station = Station()
    station.add_component(dac)
    station.add_component(dmm)

    measurement = Measurement(experiment, station)
    measurement.register_parameter(dac.ch1)

    with measurement.run(snapshot_at_end=False) as data_saver:
        data_saver.add_result((dac.ch1, 7))

    assert data_saver.dataset.snapshot is not None
    assert data_saver.dataset.end_snapshot is None
    assert "end_snapshot" not in data_saver.dataset.metadata


def test_end_snapshot_can_be_disabled_by_config(experiment, dac, dmm) -> None:
    station = Station()
    station.add_component(dac)

    measurement = Measurement(experiment, station)
    measurement.register_parameter(dac.ch1)

    original = qc.config.dataset.snapshot_at_end
    qc.config.dataset.snapshot_at_end = False
    try:
        with measurement.run() as data_saver:
            data_saver.add_result((dac.ch1, 7))
    finally:
        qc.config.dataset.snapshot_at_end = original

    assert data_saver.dataset.end_snapshot is None


def test_add_end_snapshot_does_not_overwrite(experiment, dac) -> None:
    measurement = Measurement(experiment)
    measurement.register_parameter(dac.ch1)

    with measurement.run() as data_saver:
        data_saver.add_result((dac.ch1, 7))

    dataset = data_saver.dataset
    original = dataset._end_snapshot_raw
    assert original is not None

    dataset.add_end_snapshot(json.dumps({"station": {"parameters": {}}}))
    assert dataset._end_snapshot_raw == original

    dataset.add_end_snapshot(
        json.dumps({"station": {"parameters": {}}}), overwrite=True
    )
    assert dataset.end_snapshot == {"station": {"parameters": {}}}


def test_diff_start_end_snapshot(experiment, dac, dmm) -> None:
    station = Station()
    station.add_component(dac)
    station.add_component(dmm)

    dac.ch1(1)
    dac.ch2(2)

    measurement = Measurement(experiment, station)
    measurement.register_parameter(dac.ch1)

    with measurement.run() as data_saver:
        data_saver.add_result((dac.ch1, 7))
        dac.ch1(10)

    dataset = data_saver.dataset

    diff = diff_start_end_snapshot(dataset)
    assert diff.changed[("dummy_dac", "ch1")] == (1, 10)
    assert ("dummy_dac", "ch2") not in diff.changed
    assert diff.left_only == {}
    assert diff.right_only == {}

    # the same diff can be obtained from the run id
    diff_by_id = diff_start_end_snapshot_by_id(dataset.run_id)
    assert diff_by_id == diff


def test_diff_start_end_snapshot_raises_without_end_snapshot(experiment, dac) -> None:
    measurement = Measurement(experiment)
    measurement.register_parameter(dac.ch1)

    with measurement.run(snapshot_at_end=False) as data_saver:
        data_saver.add_result((dac.ch1, 7))

    with pytest.raises(RuntimeError, match="end of the measurement is empty"):
        diff_start_end_snapshot(data_saver.dataset)


def test_format_parameter_diff() -> None:
    diff = ParameterDiff(
        left_only={"a": 1},
        right_only={("inst", "b"): 2},
        changed={("inst", "c"): (3, 4)},
    )

    formatted = format_parameter_diff(diff, "start", "end")
    assert formatted == (
        "Changed parameters (start -> end):\n"
        "  inst.c: 3 -> 4\n"
        "Parameters only in start:\n"
        "  a: 1\n"
        "Parameters only in end:\n"
        "  inst.b: 2"
    )

    assert str(diff) == format_parameter_diff(diff)

    empty = ParameterDiff(left_only={}, right_only={}, changed={})
    assert str(empty) == "No differences between the two snapshots."
