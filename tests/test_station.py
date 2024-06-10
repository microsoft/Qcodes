import json
import os
import tempfile
import warnings
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

import pytest
from ruamel.yaml import YAML

import qcodes
from qcodes import validators
from qcodes.instrument import Instrument
from qcodes.instrument_drivers.mock_instruments import (
    DummyChannelInstrument,
    DummyChannelOnlyInstrument,
    DummyInstrument,
)
from qcodes.monitor import Monitor
from qcodes.parameters import DelegateParameter, Parameter
from qcodes.station import SCHEMA_PATH, Station, ValidationWarning, update_config_schema
from qcodes.utils import NumpyJSONEncoder, get_qcodes_path

from .common import DummyComponent


@pytest.fixture(autouse=True)
def set_default_station_to_none_automatically():
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


@pytest.fixture(autouse=True)
def treat_validation_warning_as_error():
    warnings.simplefilter("error", ValidationWarning)
    yield
    warnings.simplefilter("default", ValidationWarning)


def test_station() -> None:
    bob = DummyInstrument("bob", gates=["one"])
    station = Station(bob)

    assert ["bob"] == list(station.components.keys())
    assert bob == station.components["bob"]

    assert station == station.default
    assert station == Station.default


def test_station_getitem() -> None:
    bob = DummyInstrument("bob", gates=["one"])
    station = Station(bob)

    assert bob == station["bob"]

    with pytest.raises(KeyError, match="bobby"):
        _ = station.components["bobby"]


def test_station_delegated_attributes() -> None:
    bob = DummyInstrument("bob", gates=["one"])
    station = Station(bob)

    assert bob == station.bob

    with pytest.raises(
        AttributeError,
        match="'Station' object and its delegates have no attribute 'bobby'",
    ):
        _ = station.bobby


def test_add_component() -> None:
    bob = DummyInstrument("bob", gates=["one"])
    station = Station()
    station.add_component(bob, "bob")

    assert ["bob"] == list(station.components.keys())
    assert bob == station.components["bob"]


def test_add_component_without_specifying_name() -> None:
    """
    Test that station looks for 'name' attribute in the component and uses it
    """
    bob = DummyInstrument("bob", gates=["one"])
    assert hasattr(bob, "name")
    assert "bob" == bob.name

    station = Station()
    station.add_component(bob)

    assert ["bob"] == list(station.components.keys())
    assert bob == station.components["bob"]


def test_add_component_with_no_name() -> None:
    """
    Test that station comes up with a name for components without 'name'
    attribute
    """
    bob = {"name", "bob"}
    station = Station()
    station.add_component(bob)  # type: ignore[arg-type]

    assert ["component0"] == list(station.components.keys())
    assert bob == station.components["component0"]  # type: ignore[comparison-overlap]

    jay = {"name", "jay"}
    station.add_component(jay)  # type: ignore[arg-type]

    assert ["component0", "component1"] == list(station.components.keys())
    assert jay == station.components["component1"]  # type: ignore[comparison-overlap]


def test_remove_component() -> None:
    bob = DummyInstrument("bob", gates=["one"])
    station = Station()
    station.add_component(bob, "bob")

    assert ["bob"] == list(station.components.keys())
    assert bob == station.components["bob"]

    bob2 = station.remove_component("bob")

    with pytest.raises(KeyError, match="bob"):
        _ = station.components["bob"]
    assert bob == bob2

    with pytest.raises(KeyError, match="Component bobby is not part of the station"):
        _ = station.remove_component("bobby")


def test_close_all_registered_instruments() -> None:
    names = [f"some_name_{i}" for i in range(10)]
    instrs = [Instrument(name=name) for name in names]
    st = Station(*instrs)
    for name in names:
        assert name in Instrument._all_instruments
    st.close_all_registered_instruments()
    for name in names:
        assert name not in Instrument._all_instruments


def test_snapshot() -> None:
    station = Station()

    empty_snapshot = station.snapshot()
    assert {
        "instruments": {},
        "parameters": {},
        "components": {},
        "config": None,
    } == empty_snapshot

    instrument = DummyInstrument("instrument", gates=["one"])
    station.add_component(instrument)
    instrument_snapshot = instrument.snapshot()

    parameter = Parameter("parameter", label="Label", unit="m")
    station.add_component(parameter)
    parameter_snapshot = parameter.snapshot()

    excluded_parameter = Parameter("excluded_parameter", snapshot_exclude=True)
    station.add_component(excluded_parameter)

    component = DummyComponent("component")
    component.metadata["smth"] = "in the way she moves"
    station.add_component(component)
    component_snapshot = component.snapshot()

    snapshot = station.snapshot()

    assert isinstance(snapshot, dict)
    assert [
        "instruments",
        "parameters",
        "components",
        "config",
    ] == list(snapshot.keys())

    assert ["instrument"] == list(snapshot["instruments"].keys())
    assert instrument_snapshot == snapshot["instruments"]["instrument"]

    # the list should not contain the excluded parameter
    assert ["parameter"] == list(snapshot["parameters"].keys())
    assert parameter_snapshot == snapshot["parameters"]["parameter"]

    assert ["component"] == list(snapshot["components"].keys())
    assert component_snapshot == snapshot["components"]["component"]


def test_station_after_instrument_is_closed() -> None:
    """
    Test that station is aware of the fact that its components could be
    removed within the lifetime of the station. Here we instantiate an
    instrument, add it to a station, then close the instrument, and then
    perform an action on the station to ensure that the closed instrument
    does not break the work of the station object.
    """
    bob = DummyInstrument("bob", gates=["one"])

    station = Station(bob)

    assert bob == station["bob"]

    bob.close()

    # 'bob' is closed, but it is still part of the station
    assert bob == station["bob"]

    # check that snapshot method executes without exceptions
    snapshot = station.snapshot()

    # check that 'bob's snapshot is not here (because 'bob' is closed,
    # hence it was ignored, and even removed from the station by
    # `snapshot_base` method)
    assert {
        "instruments": {},
        "parameters": {},
        "components": {},
        "config": None,
    } == snapshot

    # check that 'bob' has been removed from the station
    with pytest.raises(KeyError, match="bob"):
        _ = station.components["bob"]

    # check that 'bob' has been removed from the station, again
    with pytest.raises(KeyError, match="Component bob is not part of the station"):
        station.remove_component("bob")


def test_update_config_schema() -> None:
    update_config_schema()
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    assert len(schema["definitions"]["instruments"]["enum"]) > 1


@contextmanager
def config_file_context(file_content):
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = Path(tmpdirname, "station_config.yaml")
        with filename.open("w") as f:
            f.write(file_content)
        yield str(filename)


@contextmanager
def config_files_context(file_content1, file_content2):
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename1 = Path(tmpdirname, "station_config1.yaml")
        with filename1.open("w") as f:
            f.write(file_content1)
        filename2 = Path(tmpdirname, "station_config2.yaml")
        with filename2.open("w") as f:
            f.write(file_content2)
        yield [str(filename1), str(filename2)]


@pytest.fixture(name="example_station_config")
def _make_example_station_config():
    """
    Returns path to temp yaml file with station config.
    """
    sims_path = get_qcodes_path("instrument", "sims")
    test_config = f"""
instruments:
  lakeshore:
    type: qcodes.instrument_drivers.Lakeshore.Model_336.Model_336
    enable_forced_reconnect: true
    address: GPIB::2::INSTR
    init:
      visalib: '{sims_path}lakeshore_model336.yaml@sim'
  mock_dac:
    type: qcodes.instrument_drivers.mock_instruments.DummyInstrument
    enable_forced_reconnect: true
    init:
      gates: {{"ch1", "ch2"}}
    parameters:
      ch1:
        monitor: true
  mock_dac2:
    type: qcodes.instrument_drivers.mock_instruments.DummyInstrument
    """
    with config_file_context(test_config) as filename:
        yield filename


def test_dynamic_reload_of_file(example_station_config) -> None:
    st = Station(config_file=example_station_config)
    mock_dac = st.load_instrument("mock_dac")
    assert "ch1" in mock_dac.parameters
    with open(example_station_config) as f:
        filedata = f.read().replace("ch1", "gate1")
    with open(example_station_config, "w") as f:
        f.write(filedata)
    mock_dac = st.load_instrument("mock_dac")
    assert "ch1" not in mock_dac.parameters
    assert "gate1" in mock_dac.parameters


def station_from_config_str(config: str) -> Station:
    st = Station(config_file=None)
    st.load_config(config)
    return st


def station_config_has_been_loaded(st: Station) -> bool:
    return st.config is not None


@pytest.fixture(name="example_station")
def _make_example_station(example_station_config):
    return Station(config_file=example_station_config)


# instrument loading related tests
def test_station_config_path_resolution(example_station_config) -> None:
    config = qcodes.config["station"]

    # There is no default yaml file present that defines a station
    # so we expect the station config not to be loaded.
    assert not station_config_has_been_loaded(Station())

    path = Path(example_station_config)
    config["default_file"] = str(path)
    # Now the default file with the station configuration is specified, and
    # this file exists, hence we expect the Station to have the station
    # configuration loaded upon initialization.
    assert station_config_has_been_loaded(Station())

    config["default_file"] = path.name
    config["default_folder"] = str(path.parent)
    # Here the default_file setting contains only the file name, and the
    # default_folder contains the path to the folder where this file is
    # located, hence we again expect that the station configuration is loaded
    # upon station initialization.
    assert station_config_has_been_loaded(Station())

    config["default_file"] = "random.yml"
    config["default_folder"] = str(path.parent)
    # In this case, the station configuration file specified in the qcodes
    # config does not exist, hence the initialized station is not expected to
    # have station configuration loaded.
    assert not station_config_has_been_loaded(Station())

    config["default_file"] = str(path)
    config["default_folder"] = r"C:\SomeOtherFolder"
    # In this case, the default_file setting of the qcodes config contains
    # absolute path to the station configuration file, while the default_folder
    # setting is set to some non-existent folder.
    # In this situation, the value of the default_folder will be ignored,
    # but because the file specified in default_file setting exists,
    # the station will be initialized with the loaded configuration.
    assert station_config_has_been_loaded(Station())

    config["default_file"] = None
    config["default_folder"] = str(path.parent)
    # When qcodes config has only the default_folder setting specified to an
    # existing folder, and default_file setting is not specified, then
    # passing the name of a station configuration file, that exists in that
    # default_folder, as an argument to the Station is expected to result
    # in a station with loaded configuration.
    assert station_config_has_been_loaded(Station(config_file=path.name))

    config["default_file"] = None
    config["default_folder"] = None
    # In case qcodes config does not have default_file and default_folder
    # settings specified, passing an absolute file path as an argument to the
    # station is expected to result in a station with loaded configuration.
    assert station_config_has_been_loaded(Station(config_file=str(path)))


def test_station_configuration_is_a_component_of_station(example_station) -> None:
    assert station_config_has_been_loaded(example_station)


def test_station_config_can_be_loaded_from_snapshot(example_station) -> None:
    assert station_config_has_been_loaded(example_station)
    # ensure that we can correctly dump config which is a subclass of UserDict
    configdump = json.dumps(example_station.config, cls=NumpyJSONEncoder)
    # as this is now a regular dict we can load it back
    loaded_config = json.loads(configdump)
    # now lets ensure that we can recreate the
    # station from the loaded config
    # first we need to get a yaml repr of the data
    yaml = YAML()
    with StringIO() as output:
        yaml.dump(loaded_config, output)
        yaml_repr = output.getvalue()
    # which we can then reload into the station
    new_station = Station(default=False)
    new_station.load_config(yaml_repr)
    assert example_station.config == new_station.config


@pytest.fixture
def simple_mock_station():
    yield station_from_config_str(
        """
instruments:
  mock:
    type: qcodes.instrument_drivers.mock_instruments.DummyInstrument
        """
    )


def test_simple_mock_config(simple_mock_station) -> None:
    st = simple_mock_station
    assert station_config_has_been_loaded(st)
    assert hasattr(st, "load_mock")
    mock_snapshot = st.snapshot()["config"]["instruments"]["mock"]
    assert (
        mock_snapshot["type"]
        == "qcodes.instrument_drivers.mock_instruments.DummyInstrument"
    )
    assert "mock" in st.config["instruments"]


def test_simple_mock_load_mock(simple_mock_station) -> None:
    st = simple_mock_station
    mock = st.load_mock()
    assert isinstance(mock, DummyInstrument)
    assert mock.name == "mock"
    assert st.components["mock"] is mock


def test_simple_mock_load_instrument(simple_mock_station) -> None:
    st = simple_mock_station
    mock = st.load_instrument("mock")
    assert isinstance(mock, DummyInstrument)
    assert mock.name == "mock"
    assert st.components["mock"] is mock


def test_enable_force_reconnect() -> None:
    def get_instrument_config(enable_forced_reconnect: bool | None) -> str:
        return f"""
instruments:
  mock:
    type: qcodes.instrument_drivers.mock_instruments.DummyInstrument
    {f'enable_forced_reconnect: {enable_forced_reconnect}'
        if enable_forced_reconnect is not None else ''}
    init:
      gates: {{"ch1", "ch2"}}
         """

    def assert_on_reconnect(
        *,
        use_user_cfg: bool | None,
        use_instr_cfg: bool | None,
        expect_failure: bool,
    ) -> None:
        qcodes.config["station"]["enable_forced_reconnect"] = use_user_cfg
        st = station_from_config_str(get_instrument_config(use_instr_cfg))
        st.load_instrument("mock")
        if expect_failure:
            with pytest.raises(KeyError) as excinfo:
                st.load_instrument("mock")
            assert "Another instrument has the name: mock" in str(excinfo.value)
        else:
            st.load_instrument("mock")
        Instrument.close_all()

    for use_user_cfg in [None, True, False]:
        assert_on_reconnect(
            use_user_cfg=use_user_cfg, use_instr_cfg=False, expect_failure=True
        )
        assert_on_reconnect(
            use_user_cfg=use_user_cfg, use_instr_cfg=True, expect_failure=False
        )

    assert_on_reconnect(use_user_cfg=True, use_instr_cfg=None, expect_failure=False)
    assert_on_reconnect(use_user_cfg=False, use_instr_cfg=None, expect_failure=True)
    assert_on_reconnect(use_user_cfg=None, use_instr_cfg=None, expect_failure=True)


def test_revive_instance() -> None:
    st = station_from_config_str(
        """
instruments:
  mock:
    type: qcodes.instrument_drivers.mock_instruments.DummyInstrument
    enable_forced_reconnect: true
    init:
      gates: {"ch1"}
    """
    )
    mock = st.load_instrument("mock")
    mock2 = st.load_instrument("mock")
    assert mock is not mock2
    assert mock is not st.mock
    assert mock2 is st.mock

    mock3 = st.load_instrument("mock", revive_instance=True)
    assert mock3 == mock2
    assert mock3 == st.mock


def test_init_parameters() -> None:
    st = station_from_config_str(
        """
instruments:
  mock:
    type: qcodes.instrument_drivers.mock_instruments.DummyInstrument
    enable_forced_reconnect: true
    init:
      gates: {"ch1", "ch2"}
    """
    )
    mock = st.load_instrument("mock")
    for ch in ["ch1", "ch2"]:
        assert ch in mock.parameters.keys()
    assert len(mock.parameters) == 4  # there is also IDN and a fixed param

    # Overwrite parameter
    mock = st.load_instrument("mock", gates=["TestGate"])
    assert "TestGate" in mock.parameters.keys()
    assert len(mock.parameters) == 3  # there is also IDN and a fixed param
    # test address
    sims_path = get_qcodes_path("instrument", "sims")
    st = station_from_config_str(
        f"""
instruments:
  lakeshore:
    type: qcodes.instrument_drivers.Lakeshore.Model_336.Model_336
    enable_forced_reconnect: true
    address: GPIB::2::INSTR
    init:
      visalib: '{sims_path}lakeshore_model336.yaml@sim'
    """
    )
    st.load_instrument("lakeshore")


def test_name_init_kwarg(simple_mock_station) -> None:
    # special case of `name` as kwarg
    st = simple_mock_station
    mock = st.load_instrument("mock", name="test")
    assert mock.name == "test"
    assert st.components["test"] is mock


def test_name_specified_in_init_in_yaml_is_used() -> None:
    st = station_from_config_str(
        """
instruments:
  mock:
    type: qcodes.instrument_drivers.mock_instruments.DummyInstrument
    init:
      name: dummy
        """
    )

    mock = st.load_instrument("mock")
    assert isinstance(mock, DummyInstrument)
    assert mock.name == "dummy"
    assert st.components["dummy"] is mock


class InstrumentWithNameAsNotFirstArgument(Instrument):
    def __init__(self, first_arg, name):
        super().__init__(name)
        self._first_arg = first_arg


def test_able_to_load_instrument_with_name_argument_not_being_the_first() -> None:
    st = station_from_config_str(
        """
instruments:
  name_goes_second:
    type: tests.test_station.InstrumentWithNameAsNotFirstArgument
        """
    )

    instr = st.load_instrument("name_goes_second", first_arg=42)
    assert isinstance(instr, InstrumentWithNameAsNotFirstArgument)
    assert instr.name == "name_goes_second"
    assert st.components["name_goes_second"] is instr


def test_setup_alias_parameters() -> None:
    st = station_from_config_str(
        """
instruments:
  mock:
    type: qcodes.instrument_drivers.mock_instruments.DummyInstrument
    enable_forced_reconnect: true
    init:
      gates: {"ch1"}
    parameters:
      ch1:
        unit: mV
        label: main gate
        scale: 2
        offset: 1
        limits: [-10, 10]
        alias: gate_a
        initial_value: 9

    """
    )
    mock = st.load_instrument("mock")
    p = getattr(mock, "gate_a")
    assert isinstance(p, Parameter)
    assert p.unit == "mV"
    assert p.label == "main gate"
    assert p.scale == 2
    assert p.offset == 1
    assert isinstance(p.vals, validators.Numbers)
    assert str(p.vals) == "<Numbers -10<=v<=10>"
    assert p() == 9
    mock.ch1(1)
    assert p() == 1
    p(3)
    assert mock.ch1() == 3
    assert p.raw_value == 7
    assert mock.ch1.raw_value == 7


def test_setup_delegate_parameters() -> None:
    st = station_from_config_str(
        """
instruments:
  mock:
    type: qcodes.instrument_drivers.mock_instruments.DummyInstrument
    enable_forced_reconnect: true
    init:
      gates: {"ch1"}
    parameters:
      ch1:
        unit: V
        label: ch1
        scale: 1
        offset: 0
        limits: [-10, 10]
    add_parameters:
      gate_a:
        source: ch1
        unit: mV
        label: main gate
        scale: 2
        offset: 1
        limits: [-6.0 , 6.]
        initial_value: 2

    """
    )
    mock = st.load_instrument("mock")
    p = getattr(mock, "gate_a")
    assert isinstance(p, DelegateParameter)
    assert p.unit == "mV"
    assert p.label == "main gate"
    assert p.scale == 2
    assert p.offset == 1
    assert isinstance(p.vals, validators.Numbers)
    assert str(p.vals) == "<Numbers -6.0<=v<=6.0>"
    assert p() == 2
    assert mock.ch1.unit == "V"
    assert mock.ch1.label == "ch1"
    assert mock.ch1.scale == 1
    assert mock.ch1.offset == 0
    assert isinstance(p.vals, validators.Numbers)
    assert str(mock.ch1.vals) == "<Numbers -10<=v<=10>"
    assert mock.ch1() == 5
    mock.ch1(7)
    assert p() == 3
    assert p.raw_value == 7
    assert mock.ch1.raw_value == 7
    assert json.dumps(mock.ch1.snapshot()) == json.dumps(
        p.snapshot()["source_parameter"]
    )


def test_channel_instrument() -> None:
    """Test that parameters from instrument's submodule also get configured correctly"""
    st = station_from_config_str(
        """
instruments:
  mock:
    type: qcodes.instrument_drivers.mock_instruments.DummyChannelInstrument
    enable_forced_reconnect: true
    parameters:
      A.temperature:
        unit: mK
    add_parameters:
      T:
        source: A.temperature
      A.voltage:
        source: A.temperature
    """
    )
    mock = st.load_instrument("mock")
    assert mock.A.temperature.unit == "mK"
    assert mock.T.unit == "mK"
    assert mock.A.voltage.source is mock.A.temperature


def test_setting_channel_parameter() -> None:
    st = station_from_config_str(
        """
instruments:
  mock:
    type: qcodes.instrument_drivers.mock_instruments.DummyChannelInstrument
    parameters:
      channels.temperature:
          initial_value: 10
    """
    )
    mock = st.load_instrument("mock")
    assert mock.channels.temperature() == (10,) * 6


def test_monitor_not_loaded_by_default(example_station_config) -> None:
    st = Station(config_file=example_station_config)
    st.load_instrument("mock_dac")
    assert Monitor.running is None


def test_monitor_loaded_if_specified(
    example_station_config, request: pytest.FixtureRequest
) -> None:
    st = Station(config_file=example_station_config, use_monitor=True)
    st.load_instrument("mock_dac")
    assert Monitor.running is not None
    request.addfinalizer(Monitor.running.stop)
    assert len(Monitor.running._parameters) == 1
    assert Monitor.running._parameters[0].name == "ch1"


def test_monitor_loaded_by_default_if_in_config(
    example_station_config, request: pytest.FixtureRequest
) -> None:
    qcodes.config["station"]["use_monitor"] = True
    st = Station(config_file=example_station_config)
    st.load_instrument("mock_dac")
    assert Monitor.running is not None
    request.addfinalizer(Monitor.running.stop)
    assert len(Monitor.running._parameters) == 1
    assert Monitor.running._parameters[0].name == "ch1"


def test_monitor_not_loaded_if_specified(example_station_config) -> None:
    st = Station(config_file=example_station_config, use_monitor=False)
    st.load_instrument("mock_dac")
    assert Monitor.running is None


def test_config_validation_failure() -> None:
    with pytest.raises(ValidationWarning):
        station_from_config_str(
            """
instruments:
  mock:
    driver: qcodes.instrument_drivers.mock_instruments.DummyInstrument
invalid_keyword:
  more_errors: 42
        """
        )


def test_config_validation_failure_on_file() -> None:
    with pytest.raises(ValidationWarning):
        test_config = """
instruments:
  mock:
    driver: qcodes.instrument_drivers.mock_instruments.DummyInstrument
invalid_keyword:
  more_errors: 42
    """
        with config_file_context(test_config) as filename:
            Station(config_file=filename)


def test_config_validation_comprehensive_config() -> None:
    Station(
        config_file=os.path.join(
            get_qcodes_path(), "dist", "tests", "station", "example.station.yaml"
        )
    )


def test_load_all_instruments_raises_on_both_only_names_and_only_types_passed(
    example_station,
) -> None:
    with pytest.raises(
        ValueError,
        match="It is an error to supply both ``only_names`` "
        "and ``only_types`` arguments.",
    ):
        example_station.load_all_instruments(only_names=(), only_types=())


def test_load_all_instruments_no_args(example_station) -> None:
    all_instruments_in_config = {"lakeshore", "mock_dac", "mock_dac2"}

    loaded_instruments = example_station.load_all_instruments()

    assert set(loaded_instruments) == all_instruments_in_config

    for instrument in all_instruments_in_config:
        assert instrument in example_station.components
        assert Instrument.exist(instrument)


def test_load_all_instruments_only_types(example_station) -> None:
    all_dummy_instruments = {"mock_dac", "mock_dac2"}

    loaded_instruments = example_station.load_all_instruments(
        only_types=("DummyInstrument",)
    )

    assert set(loaded_instruments) == all_dummy_instruments

    for instrument in all_dummy_instruments:
        assert instrument in example_station.components
        assert Instrument.exist(instrument)

    other_instruments = (
        set(example_station.config["instruments"].keys()) - all_dummy_instruments
    )

    for instrument in other_instruments:
        assert instrument not in example_station.components
        assert not Instrument.exist(instrument)


def test_load_all_instruments_only_names(example_station) -> None:
    instruments_to_load = {"lakeshore", "mock_dac"}

    loaded_instruments = example_station.load_all_instruments(
        only_names=instruments_to_load
    )

    assert set(loaded_instruments) == instruments_to_load

    for instrument in loaded_instruments:
        assert instrument in example_station.components
        assert Instrument.exist(instrument)

    other_instruments = (
        set(example_station.config["instruments"].keys()) - instruments_to_load
    )

    for instrument in other_instruments:
        assert instrument not in example_station.components
        assert not Instrument.exist(instrument)


def test_load_all_instruments_without_config_raises() -> None:
    station = Station()
    with pytest.raises(ValueError, match="Station has no config"):
        station.load_all_instruments()  # type: ignore[call-overload]


def test_station_config_created_with_multiple_config_files() -> None:
    test_config1 = """
        instruments:
          mock_dac1:
            type: qcodes.instrument_drivers.mock_instruments.DummyInstrument
            enable_forced_reconnect: true
            init:
              gates: {{"ch1", "ch2"}}
            parameters:
              ch1:
                monitor: true
    """
    test_config2 = """
        instruments:
          mock_dac2:
            type: qcodes.instrument_drivers.mock_instruments.DummyInstrument
    """
    with config_files_context(test_config1, test_config2) as file_list:
        assert station_config_has_been_loaded(Station(config_file=file_list))


def test_get_component_by_name() -> None:
    instr = DummyChannelInstrument(name="dummy")
    instr2 = DummyChannelOnlyInstrument(name="some_other_dummy")
    param = Parameter(name="param", set_cmd=None, get_cmd=None)
    station = Station(instr, instr2, param)

    assert station.get_component("dummy") is instr
    assert station.get_component("dummy_A") is instr.A
    assert station.get_component("dummy_ChanA") is instr.A
    assert station.get_component("dummy_A_temperature") is instr.A.temperature
    assert station.get_component("dummy_ChanA_temperature") is instr.A.temperature

    assert station.get_component("some_other_dummy") is instr2
    assert station.get_component("some_other_dummy_ChanA_a") is instr2.channels[0]
    assert (
        station.get_component("some_other_dummy_ChanA_a_temperature")
        is instr2.channels[0].temperature
    )

    assert station.get_component("param") is param


def test_get_wrong_component_by_name_raises() -> None:
    instr = DummyChannelInstrument(name="dummy")
    param = Parameter(name="param", set_cmd=None, get_cmd=None)
    station = Station(instr, param)

    with pytest.raises(KeyError, match="Component notdummy is not part of the station"):
        _ = station.get_component("notdummy")

    with pytest.raises(
        KeyError, match="Found component dummy but could not match notachannel part"
    ):
        _ = station.get_component("dummy_notachannel")

    with pytest.raises(
        KeyError,
        match=(
            "Found component dummy_ChanA but could "
            "not match temperature_parameter part"
        ),
    ):
        _ = station.get_component("dummy_ChanA_temperature_parameter")

    with pytest.raises(
        KeyError, match="Found component param but this has no sub-component foo."
    ):
        _ = station.get_component("param_foo")


def test_component_by_name_with_underscore_in_name() -> None:
    instr = DummyChannelInstrument(name="dum_my")
    station = Station(instr)
    assert station.get_component("dum_my") is instr
    assert station.get_component("dum_my_A") is instr.A
    assert station.get_component("dum_my_ChanA") is instr.A
    assert station.get_component("dum_my_A_temperature") is instr.A.temperature
    assert station.get_component("dum_my_ChanA_temperature") is instr.A.temperature
    assert station.get_component("dum_my_ChanA_log_my_name") is instr.A.log_my_name
