import json

import pytest

import qcodes as qc
from qcodes.tests.instrument_mocks import DummyInstrument
from qcodes.dataset.measurements import Measurement

# pylint: disable=unused-import
from qcodes.tests.dataset.temporary_databases import experiment, empty_temp_db


@pytest.fixture  # scope is "function" per default
def dac():
    dac = DummyInstrument('dummy_dac', gates=['ch1', 'ch2'])
    yield dac
    dac.close()


@pytest.fixture
def dmm():
    dmm = DummyInstrument('dummy_dmm', gates=['v1', 'v2'])
    yield dmm
    dmm.close()


@pytest.mark.parametrize("pass_station", (True, False))
def test_station_snapshot_during_measurement(experiment, dac, dmm,
                                             pass_station):
    station = qc.Station()
    station.add_component(dac)
    station.add_component(dmm, 'renamed_dmm')

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

    with measurement.run() as data_saver:
        data_saver.add_result((dac.ch1, 7), (dmm.v1, 5))

    # 1. Test `get_metadata('snapshot')` method

    json_snapshot_from_dataset = data_saver.dataset.get_metadata('snapshot')
    snapshot_from_dataset = json.loads(json_snapshot_from_dataset)

    expected_snapshot = {'station': snapshot_of_station}
    assert expected_snapshot == snapshot_from_dataset

    # 2. Test `snapshot_raw` property

    assert json_snapshot_from_dataset == data_saver.dataset.snapshot_raw

    # 3. Test `snapshot` property

    assert expected_snapshot == data_saver.dataset.snapshot
