import pytest
import tempfile
import json
from pathlib import Path
from typing import Union

import qcodes
import qcodes.utils.validators as validators
from qcodes.utils.helpers import get_qcodes_path
from qcodes.instrument.parameter import DelegateParameter
from qcodes import Instrument
from qcodes.station import Station
from qcodes.tests.instrument_mocks import (
    DummyInstrument)
from qcodes.tests.test_combined_par import DumyPar
from qcodes.instrument.parameter import Parameter
from qcodes.tests.test_config import default_config

@pytest.fixture(autouse=True)
def use_default_config():
    with default_config():
        yield

@pytest.fixture(autouse=True)
def set_default_station_to_none():
    """Makes sure that after startup and teardown there is no default station"""
    Station.default = None
    yield
    Station.default = None


@pytest.fixture(autouse=True)
def close_all_instruments():
    """Makes sure that after startup and teardown all instruments are closed"""
    Instrument.close_all()
    yield
    Instrument.close_all()


def test_station():
    bob = DummyInstrument('bob', gates=['one'])
    station = Station(bob)

    assert ['bob'] == list(station.components.keys())
    assert bob == station.components['bob']

    assert station == station.default
    assert station == Station.default


def test_station_getitem():
    bob = DummyInstrument('bob', gates=['one'])
    station = Station(bob)

    assert bob == station['bob']

    with pytest.raises(KeyError, match='bobby'):
        _ = station.components['bobby']


def test_station_delegated_attributes():
    bob = DummyInstrument('bob', gates=['one'])
    station = Station(bob)

    assert bob == station.bob

    with pytest.raises(AttributeError, match="'Station' object and its "
                                             "delegates have no attribute "
                                             "'bobby'"):
        _ = station.bobby


def test_add_component():
    bob = DummyInstrument('bob', gates=['one'])
    station = Station()
    station.add_component(bob, 'bob')

    assert ['bob'] == list(station.components.keys())
    assert bob == station.components['bob']


def test_add_component_without_specifying_name():
    """
    Test that station looks for 'name' attribute in the component and uses it
    """
    bob = DummyInstrument('bob', gates=['one'])
    assert hasattr(bob, 'name')
    assert 'bob' == bob.name

    station = Station()
    station.add_component(bob)

    assert ['bob'] == list(station.components.keys())
    assert bob == station.components['bob']


def test_add_component_with_no_name():
    """
    Test that station comes up with a name for components without 'name'
    attribute
    """
    bob = {'name', 'bob'}
    station = Station()
    station.add_component(bob)

    assert ['component0'] == list(station.components.keys())
    assert bob == station.components['component0']

    jay = {'name', 'jay'}
    station.add_component(jay)

    assert ['component0', 'component1'] == list(station.components.keys())
    assert jay == station.components['component1']


def test_remove_component():
    bob = DummyInstrument('bob', gates=['one'])
    station = Station()
    station.add_component(bob, 'bob')

    assert ['bob'] == list(station.components.keys())
    assert bob == station.components['bob']

    bob2 = station.remove_component('bob')

    with pytest.raises(KeyError, match='bob'):
        _ = station.components['bob']
    assert bob == bob2

    with pytest.raises(KeyError, match='Component bobby is not part of the '
                                       'station'):
        _ = station.remove_component('bobby')


def test_snapshot():
    station = Station()

    empty_snapshot = station.snapshot()
    assert {'instruments': {},
            'parameters': {},
            'components': {},
            'default_measurement': []
            } == empty_snapshot

    instrument = DummyInstrument('instrument', gates=['one'])
    station.add_component(instrument)
    instrument_snapshot = instrument.snapshot()

    parameter = Parameter('parameter', label='Label', unit='m')
    station.add_component(parameter)
    parameter_snapshot = parameter.snapshot()

    component = DumyPar('component')
    component.metadata['smth'] = 'in the way she moves'
    station.add_component(component)
    component_snapshot = component.snapshot()

    snapshot = station.snapshot()

    assert isinstance(snapshot, dict)
    assert ['instruments',
            'parameters',
            'components',
            'default_measurement'
            ] == list(snapshot.keys())

    assert ['instrument'] == list(snapshot['instruments'].keys())
    assert instrument_snapshot == snapshot['instruments']['instrument']

    assert ['parameter'] == list(snapshot['parameters'].keys())
    assert parameter_snapshot == snapshot['parameters']['parameter']

    assert ['component'] == list(snapshot['components'].keys())
    assert component_snapshot == snapshot['components']['component']

    assert [] == snapshot['default_measurement']


def test_station_after_instrument_is_closed():
    """
    Test that station is aware of the fact that its components could be
    removed within the lifetime of the station. Here we instantiate an
    instrument, add it to a station, then close the instrument, and then
    perform an action on the station to ensure that the closed instrument
    does not break the work of the station object.
    """
    bob = DummyInstrument('bob', gates=['one'])

    station = Station(bob)

    assert bob == station['bob']

    bob.close()

    # 'bob' is closed, but it is still part of the station
    assert bob == station['bob']

    # check that snapshot method executes without exceptions
    snapshot = station.snapshot()

    # check that 'bob's snapshot is not here (because 'bob' is closed,
    # hence it was ignored, and even removed from the station by
    # `snapshot_base` method)
    assert {'instruments': {},
            'parameters': {},
            'components': {},
            'default_measurement': []
            } == snapshot

    # check that 'bob' has been removed from the station
    with pytest.raises(KeyError, match='bob'):
        _ = station.components['bob']

    # check that 'bob' has been removed from the station, again
    with pytest.raises(KeyError, match='Component bob is not part of the '
                                       'station'):
        station.remove_component('bob')

@pytest.fixture
def example_station_config():
    """
    Returns path to temp yaml file with station config.
    """
    sims_path = get_qcodes_path('instrument', 'sims')
    test_config = f"""
instruments:
  lakeshore:
    driver: qcodes.instrument_drivers.Lakeshore.Model_336
    type: Model_336
    enable_forced_reconnect: true
    address: GPIB::2::65535::INSTR
    init:
      visalib: '{sims_path}lakeshore_model336.yaml@sim'
  mock_dac:
    driver: qcodes.tests.instrument_mocks
    type: DummyInstrument
    enable_forced_reconnect: true
    init:
      gates: {{"ch1", "ch2"}}
  mock_dac2:
    driver: qcodes.tests.instrument_mocks
    type: DummyInstrument
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = Path(tmpdirname, 'station_config.yaml')
        with filename.open('w') as f:
            f.write(test_config)
        yield str(filename)

def station_from_config_str(config: str) -> Station:
    st = Station(config_file=None)
    st.load_config(config)
    return st


def has_station_config_been_loaded(st: Station) -> bool:
    return "StationConfigurator" in st.components.keys()


@pytest.fixture
def example_station(example_station_config):
    return Station(config_file=example_station_config)


# instrument loading related tests
def test_station_config_path_resolution(example_station_config):
    config = qcodes.config["station_configurator"]

    assert not has_station_config_been_loaded(Station())

    path = Path(example_station_config)
    config["default_file"] = str(path)
    assert has_station_config_been_loaded(Station())

    config["default_file"] = path.name
    config["default_folder"] = str(path.parent)
    assert has_station_config_been_loaded(Station())

    config["default_file"] = 'random.yml'
    config["default_folder"] = str(path.parent)
    assert not has_station_config_been_loaded(Station())

    config["default_file"] = str(path)
    config["default_folder"] = r'C:\SomeOtherFolder'
    assert has_station_config_been_loaded(Station())

    config["default_file"] = None
    config["default_folder"] = str(path.parent)
    assert has_station_config_been_loaded(Station(config_file=path.name))

    config["default_file"] = None
    config["default_folder"] = None
    assert has_station_config_been_loaded(Station(config_file=str(path)))


def test_station_creation(example_station):
    assert "StationConfigurator" in example_station.components.keys()

SIMPLE_MOCK_CONFIG = """
instruments:
  mock:
    driver: qcodes.tests.instrument_mocks
    type: DummyInstrument
"""

@pytest.fixture
def simple_mock_station(example_station_config):
    yield station_from_config_str(SIMPLE_MOCK_CONFIG)

def test_simple_mock_config(simple_mock_station):
    st = simple_mock_station
    assert has_station_config_been_loaded(st)
    assert hasattr(st, 'load_mock')
    mock_snapshot = st.snapshot()['components']['StationConfigurator']\
        ['instruments']['mock']
    assert mock_snapshot['driver'] == "qcodes.tests.instrument_mocks"
    assert mock_snapshot['type'] == "DummyInstrument"


def test_simple_mock_load_mock(simple_mock_station):
    st = simple_mock_station
    mock = st.load_mock()
    assert isinstance(mock, DummyInstrument)
    assert mock.name == 'mock'
    assert st.components['mock'] is mock


def test_simple_mock_load_instrument(simple_mock_station):
    st = simple_mock_station
    mock = st.load_instrument('mock')
    assert isinstance(mock, DummyInstrument)
    assert mock.name == 'mock'
    assert st.components['mock'] is mock


def test_enable_force_reconnect() -> None:
    def get_instrument_config(enable_forced_reconnect: Union[bool, None]) -> str:
        return f"""
instruments:
  mock:
    driver: qcodes.tests.instrument_mocks
    type: DummyInstrument
    {f'enable_forced_reconnect: {enable_forced_reconnect}'
        if enable_forced_reconnect is not None else ''}
    init:
      gates: {{"ch1", "ch2"}}
         """

    def assert_on_reconnect(user_config_val: Union[bool, None],
                            instrument_config_val: Union[bool, None],
                            expect_failure: bool) -> None:
        qcodes.config["station_configurator"]\
            ['enable_forced_reconnect'] = user_config_val
        st = station_from_config_str(
            get_instrument_config(instrument_config_val))
        st.load_instrument('mock')
        if expect_failure:
            with pytest.raises(KeyError) as excinfo:
                st.load_instrument('mock')
                assert ("Another instrument has the name: mock"
                        in str(excinfo.value))
        else:
            st.load_instrument('mock')
        Instrument.close_all()

    for user_config_val in [None, True, False]:
        assert_on_reconnect(user_config_val, False, True)
        assert_on_reconnect(user_config_val, True, False)

    assert_on_reconnect(True, None, False)
    assert_on_reconnect(False, None, True)
    assert_on_reconnect(None, None, True)


def test_init_parameters():
    st = station_from_config_str("""
instruments:
  mock:
    driver: qcodes.tests.instrument_mocks
    type: DummyInstrument
    enable_forced_reconnect: true
    init:
      gates: {"ch1", "ch2"}
    """)
    mock = st.load_instrument('mock')
    for ch in ["ch1", "ch2"]:
        assert ch in mock.parameters.keys()
    assert len(mock.parameters) == 3  # there is also IDN

    # Overwrite parameter
    mock = st.load_instrument('mock', gates=["TestGate"])
    assert "TestGate" in mock.parameters.keys()
    assert len(mock.parameters) == 2  # there is also IDN

    # special case of `name`
    mock = st.load_instrument('mock', name='test')
    assert mock.name == 'test'
    assert st.components['test'] is mock

    # test address
    sims_path = get_qcodes_path('instrument', 'sims')
    st = station_from_config_str(f"""
instruments:
  lakeshore:
    driver: qcodes.instrument_drivers.Lakeshore.Model_336
    type: Model_336
    enable_forced_reconnect: true
    address: GPIB::2::INSTR
    init:
      visalib: '{sims_path}lakeshore_model336.yaml@sim'
    """)
    st.load_instrument('lakeshore')


def test_setup_alias_parameters():
    st = station_from_config_str("""
instruments:
  mock:
    driver: qcodes.tests.instrument_mocks
    type: DummyInstrument
    enable_forced_reconnect: true
    init:
      gates: {"ch1"}
    parameters:
      ch1:
        unit: mV
        label: main gate
        scale: 2
        offset: 1
        limits: -10, 10
        alias: gate_a
        initial_value: 9

    """)
    mock = st.load_instrument('mock')
    p = getattr(mock, 'gate_a')
    assert isinstance(p, qcodes.Parameter)
    assert p.unit == 'mV'
    assert p.label == 'main gate'
    assert p.scale == 2
    assert p.offset == 1
    assert isinstance(p.vals, validators.Numbers)
    assert str(p.vals) == '<Numbers -10.0<=v<=10.0>'
    assert p() == 9
    mock.ch1(1)
    assert p() == 1
    p(3)
    assert mock.ch1() == 3
    assert p.raw_value == 7
    assert mock.ch1.raw_value == 7

def test_setup_delegate_parameters():
    st = station_from_config_str("""
instruments:
  mock:
    driver: qcodes.tests.instrument_mocks
    type: DummyInstrument
    enable_forced_reconnect: true
    init:
      gates: {"ch1"}
    parameters:
      ch1:
        unit: V
        label: ch1
        scale: 1
        offset: 0
        limits: -10, 10
    add_parameters:
      gate_a:
        source: ch1
        unit: mV
        label: main gate
        scale: 2
        offset: 1
        limits: -6, 6
        initial_value: 2

    """)
    mock = st.load_instrument('mock')
    p = getattr(mock, 'gate_a')
    assert isinstance(p, DelegateParameter)
    assert p.unit == 'mV'
    assert p.label == 'main gate'
    assert p.scale == 2
    assert p.offset == 1
    assert isinstance(p.vals, validators.Numbers)
    assert str(p.vals) == '<Numbers -6.0<=v<=6.0>'
    assert p() == 2
    assert mock.ch1.unit == 'V'
    assert mock.ch1.label == 'ch1'
    assert mock.ch1.scale == 1
    assert mock.ch1.offset == 0
    assert isinstance(p.vals, validators.Numbers)
    assert str(mock.ch1.vals) == '<Numbers -10.0<=v<=10.0>'
    assert mock.ch1() == 5
    mock.ch1(7)
    assert p() == 3
    assert p.raw_value == 7
    assert mock.ch1.raw_value == 7
    assert (json.dumps(mock.ch1.snapshot()) ==
            json.dumps(p.snapshot()['source_parameter']))


def test_channel_instrument():
    """Test that parameters from instrument's submodule also get configured correctly"""
    st = station_from_config_str("""
instruments:
  mock:
    driver: qcodes.tests.instrument_mocks
    type: DummyChannelInstrument
    enable_forced_reconnect: true
    parameters:
      A.temperature:
        unit: mK
    add_parameters:
      T:
        source: A.temperature
    """)
    mock = st.load_instrument('mock')
    assert mock.A.temperature.unit == 'mK'
    assert mock.T.unit == 'mK'
