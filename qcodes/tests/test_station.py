import pytest

from qcodes import Instrument
from qcodes.station import Station
from qcodes.tests.instrument_mocks import DummyInstrument
from qcodes.tests.test_combined_par import DumyPar
from qcodes.instrument.parameter import Parameter


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
