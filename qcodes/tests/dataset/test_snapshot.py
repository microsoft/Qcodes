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


@pytest.mark.usefixtures("experiment")
def test_station_snapshot_during_measurement(experiment, dac, dmm):
    station = qc.Station()
    station.add_component(dac)
    station.add_component(dmm, 'renamed_dmm')

    snapshot_of_station = station.snapshot()

    measurement = Measurement(experiment, station)

    measurement.register_parameter(dac.ch1)
    measurement.register_parameter(dmm.v1, setpoints=[dac.ch1])

    with measurement.run() as data_saver:
        data_saver.add_result((dac.ch1, 7), (dmm.v1, 5))

    json_snapshot_from_dataset = data_saver.dataset.get_metadata('snapshot')
    snapshot_from_dataset = json.loads(json_snapshot_from_dataset)

    expected_snapshot = {'station': snapshot_of_station}
    assert expected_snapshot == snapshot_from_dataset
