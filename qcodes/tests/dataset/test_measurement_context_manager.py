import pytest
import tempfile
import os
from time import sleep
import json

from hypothesis import given, settings
import hypothesis.strategies as hst
import numpy as np

import qcodes as qc
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.experiment_container import new_experiment
from qcodes.tests.instrument_mocks import DummyInstrument
from qcodes.dataset.param_spec import ParamSpec
from qcodes.dataset.sqlite_base import connect, init_db
from qcodes.instrument.parameter import ArrayParameter
from qcodes.dataset.legacy_import import import_dat_file
from qcodes.dataset.data_set import load_by_id
from qcodes.dataset.database import initialise_database


@pytest.fixture(scope="function")
def empty_temp_db():
    # create a temp database for testing
    with tempfile.TemporaryDirectory() as tmpdirname:
        qc.config["core"]["db_location"] = os.path.join(tmpdirname, 'temp.db')
        qc.config["core"]["db_debug"] = True
        initialise_database()
        yield


@pytest.fixture(scope='function')
def experiment(empty_temp_db):
    e = new_experiment("test-experiment", sample_name="test-sample")
    yield e
    e.conn.close()


@pytest.fixture  # scope is "function" per default
def DAC():
    dac = DummyInstrument('dummy_dac', gates=['ch1', 'ch2'])
    yield dac
    dac.close()


@pytest.fixture
def DMM():
    dmm = DummyInstrument('dummy_dmm', gates=['v1', 'v2'])
    yield dmm
    dmm.close()


@pytest.fixture
def SpectrumAnalyzer():
    """
    Yields a DummyInstrument that holds an ArrayParameter
    """

    class Spectrum(ArrayParameter):

        def __init__(self, name, instrument):
            super().__init__(name=name,
                             shape=(1,),  # this attribute should be removed
                             label='Flower Power Spectrum',
                             unit='V/sqrt(Hz)',
                             setpoint_names=('Frequency',),
                             setpoint_units=('Hz',))

            self.npts = 100
            self.start = 0
            self.stop = 2e6
            self._instrument = instrument

        def get_raw(self):
            # This is how it should be: the setpoints are generated at the
            # time we get. But that will of course not work with the old Loop
            self.setpoints = (tuple(np.linspace(self.start, self.stop,
                                                self.npts)),)
            # not the best SA on the market; it just returns noise...
            return np.random.randn(self.npts)

    SA = DummyInstrument('dummy_SA')
    SA.add_parameter('spectrum', parameter_class=Spectrum)

    yield SA

    SA.close()


def test_register_parameter_numbers(DAC, DMM):
    """
    Test the registration of scalar QCoDeS parameters
    """

    parameters = [DAC.ch1, DAC.ch2, DMM.v1, DMM.v2]
    not_parameters = ['', 'Parameter', 0, 1.1, Measurement]

    meas = Measurement()

    for not_a_parameter in not_parameters:
        with pytest.raises(ValueError):
            meas.register_parameter(not_a_parameter)

    my_param = DAC.ch1
    meas.register_parameter(my_param)
    assert len(meas.parameters) == 1
    paramspec = meas.parameters[str(my_param)]
    assert paramspec.name == str(my_param)
    assert paramspec.label == my_param.label
    assert paramspec.unit == my_param.unit
    assert paramspec.type == 'numeric'

    # registering the same parameter twice should lead
    # to a replacement/update, but also change the
    # parameter order behind the scenes
    # (to allow us to re-register a parameter with new
    # setpoints)

    my_param.unit = my_param.unit + '/s'
    meas.register_parameter(my_param)
    assert len(meas.parameters) == 1
    paramspec = meas.parameters[str(my_param)]
    assert paramspec.name == str(my_param)
    assert paramspec.label == my_param.label
    assert paramspec.unit == my_param.unit
    assert paramspec.type == 'numeric'

    for parameter in parameters:
        with pytest.raises(ValueError):
            meas.register_parameter(my_param, setpoints=(parameter,))
        with pytest.raises(ValueError):
            meas.register_parameter(my_param, basis=(parameter,))

    meas.register_parameter(DAC.ch2)
    meas.register_parameter(DMM.v1)
    meas.register_parameter(DMM.v2)
    meas.register_parameter(my_param, basis=(DAC.ch2,),
                            setpoints=(DMM.v1, DMM.v2))

    assert list(meas.parameters.keys()) == [str(DAC.ch2),
                                            str(DMM.v1), str(DMM.v2),
                                            str(my_param)]
    paramspec = meas.parameters[str(my_param)]
    assert paramspec.name == str(my_param)
    assert paramspec.inferred_from == ', '.join([str(DAC.ch2)])
    assert paramspec.depends_on == ', '.join([str(DMM.v1), str(DMM.v2)])

    meas = Measurement()

    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DAC.ch2, setpoints=(DAC.ch1,))
    with pytest.raises(ValueError):
        meas.register_parameter(DMM.v1, setpoints=(DAC.ch2,))


def test_register_custom_parameter(DAC):
    """
    Test the registration of custom parameters
    """
    meas = Measurement()

    name = 'V_modified'
    unit = 'V^2'
    label = 'square of the voltage'

    meas.register_custom_parameter(name, label, unit)

    assert len(meas.parameters) == 1
    assert isinstance(meas.parameters[name], ParamSpec)
    assert meas.parameters[name].unit == unit
    assert meas.parameters[name].label == label
    assert meas.parameters[name].type == 'numeric'

    newunit = 'V^3'
    newlabel = 'cube of the voltage'

    meas.register_custom_parameter(name, newlabel, newunit)

    assert len(meas.parameters) == 1
    assert isinstance(meas.parameters[name], ParamSpec)
    assert meas.parameters[name].unit == newunit
    assert meas.parameters[name].label == newlabel

    with pytest.raises(ValueError):
        meas.register_custom_parameter(name, label, unit,
                                       setpoints=(DAC.ch1,))
    with pytest.raises(ValueError):
        meas.register_custom_parameter(name, label, unit,
                                       basis=(DAC.ch2,))

    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DAC.ch2)
    meas.register_custom_parameter('strange_dac')

    meas.register_custom_parameter(name, label, unit,
                                   setpoints=(DAC.ch1, str(DAC.ch2)),
                                   basis=('strange_dac',))

    assert len(meas.parameters) == 4
    parspec = meas.parameters[name]
    assert parspec.inferred_from == 'strange_dac'
    assert parspec.depends_on == ', '.join([str(DAC.ch1), str(DAC.ch2)])

    with pytest.raises(ValueError):
        meas.register_custom_parameter('double dependence',
                                       'label', 'unit', setpoints=(name,))


def test_unregister_parameter(DAC, DMM):
    """
    Test the unregistering of parameters.
    """

    DAC.add_parameter('impedance',
                      get_cmd=lambda: 5)

    meas = Measurement()

    meas.register_parameter(DAC.ch2)
    meas.register_parameter(DMM.v1)
    meas.register_parameter(DMM.v2)
    meas.register_parameter(DAC.ch1, basis=(DMM.v1, DMM.v2),
                            setpoints=(DAC.ch2,))

    with pytest.raises(ValueError):
        meas.unregister_parameter(DAC.ch2)
    with pytest.raises(ValueError):
        meas.unregister_parameter(str(DAC.ch2))
    with pytest.raises(ValueError):
        meas.unregister_parameter(DMM.v1)
    with pytest.raises(ValueError):
        meas.unregister_parameter(DMM.v2)

    meas.unregister_parameter(DAC.ch1)
    assert list(meas.parameters.keys()) == [str(DAC.ch2), str(DMM.v1),
                                            str(DMM.v2)]

    meas.unregister_parameter(DAC.ch2)
    assert list(meas.parameters.keys()) == [str(DMM.v1), str(DMM.v2)]

    not_parameters = [DAC, DMM, 0.0, 1]
    for notparam in not_parameters:
        with pytest.raises(ValueError):
            meas.unregister_parameter(notparam)

    # unregistering something not registered should silently "succeed"
    meas.unregister_parameter('totes_not_registered')
    meas.unregister_parameter(DAC.ch2)
    meas.unregister_parameter(DAC.ch2)


def test_measurement_name(experiment, DAC, DMM):

    fmt = experiment.format_string
    exp_id = experiment.exp_id

    name = 'yolo'

    meas = Measurement()
    meas.name = name

    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DMM.v1, setpoints=[DAC.ch1])

    with meas.run() as datasaver:
        run_id = datasaver.run_id
        expected_name = fmt.format(name, exp_id, run_id)
        assert datasaver.dataset.table_name == expected_name


@given(wp=hst.one_of(hst.integers(), hst.floats(allow_nan=False),
                     hst.text()))
def test_setting_write_period(empty_temp_db, wp):
    new_experiment('firstexp', sample_name='no sample')
    meas = Measurement()

    if isinstance(wp, str):
        with pytest.raises(ValueError):
            meas.write_period = wp
    elif wp < 1e-3:
        with pytest.raises(ValueError):
            meas.write_period = wp
    else:
        meas.write_period = wp
        assert meas._write_period == wp

        with meas.run() as datasaver:
            assert datasaver.write_period == wp


@given(words=hst.lists(elements=hst.text(), min_size=4, max_size=10))
def test_enter_and_exit_actions(experiment, DAC, words):

    # we use a list to check that the functions executed
    # in the correct order

    def action(lst, word):
        lst.append(word)

    meas = Measurement()
    meas.register_parameter(DAC.ch1)

    testlist = []

    splitpoint = round(len(words)/2)
    for n in range(splitpoint):
        meas.add_before_run(action, (testlist, words[n]))
    for m in range(splitpoint, len(words)):
        meas.add_after_run(action, (testlist, words[m]))

    assert len(meas.enteractions) == splitpoint
    assert len(meas.exitactions) == len(words) - splitpoint

    with meas.run() as _:
        assert testlist == words[:splitpoint]

    assert testlist == words

    meas = Measurement()

    with pytest.raises(ValueError):
        meas.add_before_run(action, 'no list!')
    with pytest.raises(ValueError):
        meas.add_after_run(action, testlist)


def test_subscriptions(experiment, DAC, DMM):

    def subscriber1(results, length, state):
        """
        A dict of all results
        """
        state[length] = results

    def subscriber2(results, length, state):
        """
        A list of all parameter values larger than 7
        """
        for res in results:
            state += [pres for pres in res if pres > 7]

    meas = Measurement(exp=experiment)
    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DMM.v1, setpoints=(DAC.ch1,))

    res_dict = {}
    lt7s = []

    meas.add_subscriber(subscriber1, state=res_dict)
    assert len(meas.subscribers) == 1
    meas.add_subscriber(subscriber2, state=lt7s)
    assert len(meas.subscribers) == 2

    meas.write_period = 0.2

    expected_list = []

    with meas.run() as datasaver:

        assert len(datasaver._dataset.subscribers) == 2
        assert res_dict == {}
        assert lt7s == []

        as_and_bs = list(zip(range(5), range(3, 8)))

        for num in range(5):

            (a, b) = as_and_bs[num]
            expected_list += [c for c in (a, b) if c > 7]
            sleep(1.2*meas.write_period)
            datasaver.add_result((DAC.ch1, a), (DMM.v1, b))
            assert lt7s == expected_list
            assert list(res_dict.keys()) == [n for n in range(1, num+2)]

    assert len(datasaver._dataset.subscribers) == 0


@settings(deadline=None, max_examples=25)
@given(N=hst.integers(min_value=2000, max_value=3000))
def test_subscriptions_getting_all_points(experiment, DAC, DMM, N):

    def sub_get_x_vals(results, length, state):
        """
        A list of all x values
        """
        state += [res[0] for res in results]

    def sub_get_y_vals(results, length, state):
        """
        A list of all y values
        """
        state += [res[1] for res in results]

    meas = Measurement(exp=experiment)
    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DMM.v1, setpoints=(DAC.ch1,))

    xvals = []
    yvals = []

    meas.add_subscriber(sub_get_x_vals, state=xvals)
    meas.add_subscriber(sub_get_y_vals, state=yvals)

    given_xvals = range(N)
    given_yvals = range(1, N+1)

    with meas.run() as datasaver:

        for x, y in zip(given_xvals, given_yvals):
            datasaver.add_result((DAC.ch1, x), (DMM.v1, y))

    assert xvals == list(given_xvals)
    assert yvals == list(given_yvals)


# There is no way around it: this test is slow. We test that write_period
# works and hence we must wait for some time to elapse. Sorry.
@settings(max_examples=5, deadline=None)
@given(breakpoint=hst.integers(min_value=1, max_value=19),
       write_period=hst.floats(min_value=0.1, max_value=1.5),
       set_values=hst.lists(elements=hst.floats(), min_size=20, max_size=20),
       get_values=hst.lists(elements=hst.floats(), min_size=20, max_size=20))
def test_datasaver_scalars(experiment, DAC, DMM, set_values, get_values,
                           breakpoint, write_period):

    no_of_runs = len(experiment)

    station = qc.Station(DAC, DMM)

    meas = Measurement(station=station)
    meas.write_period = write_period

    assert meas.write_period == write_period

    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DMM.v1, setpoints=(DAC.ch1,))

    with meas.run() as datasaver:
        for set_v, get_v in zip(set_values[:breakpoint], get_values[:breakpoint]):
            datasaver.add_result((DAC.ch1, set_v), (DMM.v1, get_v))

        assert datasaver._dataset.number_of_results == 0
        sleep(write_period * 1.1)
        datasaver.add_result((DAC.ch1, set_values[breakpoint]),
                             (DMM.v1, get_values[breakpoint]))
        assert datasaver.points_written == breakpoint + 1

    assert datasaver.run_id == no_of_runs + 1

    with meas.run() as datasaver:
        with pytest.raises(ValueError):
            datasaver.add_result((DAC.ch2, 1), (DAC.ch2, 2))
        with pytest.raises(ValueError):
            datasaver.add_result((DMM.v1, 0))

    # important cleanup, else the following tests will fail
    qc.Station.default = None

    # More assertions of setpoints, labels and units in the DB!


@settings(max_examples=10, deadline=None)
@given(N=hst.integers(min_value=2, max_value=500))
def test_datasaver_arrays(empty_temp_db, N):
    new_experiment('firstexp', sample_name='no sample')

    meas = Measurement()

    meas.register_custom_parameter(name='freqax',
                                   label='Frequency axis',
                                   unit='Hz')
    meas.register_custom_parameter(name='signal',
                                   label='qubit signal',
                                   unit='Majorana number',
                                   setpoints=('freqax',))

    with meas.run() as datasaver:
        freqax = np.linspace(1e6, 2e6, N)
        signal = np.random.randn(N)

        datasaver.add_result(('freqax', freqax), ('signal', signal))

    assert datasaver.points_written == N

    with meas.run() as datasaver:
        freqax = np.linspace(1e6, 2e6, N)
        signal = np.random.randn(N-1)

        with pytest.raises(ValueError):
            datasaver.add_result(('freqax', freqax), ('signal', signal))

    meas.register_custom_parameter(name='gate_voltage',
                                   label='Gate tuning potential',
                                   unit='V')
    meas.register_custom_parameter(name='signal',
                                   label='qubit signal',
                                   unit='Majorana flux',
                                   setpoints=('freqax', 'gate_voltage'))

    with meas.run() as datasaver:
        freqax = np.linspace(1e6, 2e6, N)
        signal = np.random.randn(N)

        datasaver.add_result(('freqax', freqax),
                             ('signal', signal),
                             ('gate_voltage', 0))

    assert datasaver.points_written == N


@settings(max_examples=5, deadline=None)
@given(N=hst.integers(min_value=5, max_value=500),
       M=hst.integers(min_value=4, max_value=250))
def test_datasaver_array_parameters(experiment, SpectrumAnalyzer, DAC, N, M):

    spectrum = SpectrumAnalyzer.spectrum

    meas = Measurement()

    meas.register_parameter(spectrum)

    assert len(meas.parameters) == 2
    assert meas.parameters[str(spectrum)].depends_on == 'dummy_SA_Frequency'
    assert meas.parameters[str(spectrum)].type == 'numeric'
    assert meas.parameters['dummy_SA_Frequency'].type == 'numeric'

    # Now for a real measurement

    meas = Measurement()

    meas.register_parameter(DAC.ch1)
    meas.register_parameter(spectrum, setpoints=[DAC.ch1])

    assert len(meas.parameters) == 3

    spectrum.npts = M

    with meas.run() as datasaver:
        for set_v in np.linspace(0, 0.01, N):
            datasaver.add_result((DAC.ch1, set_v),
                                 (spectrum, spectrum.get()))

    assert datasaver.points_written == N*M


def test_load_legacy_files_2D(experiment):
    location = 'fixtures/2018-01-17/#002_2D_test_15-43-14'
    dir = os.path.dirname(__file__)
    full_location = os.path.join(dir, location)
    run_ids = import_dat_file(full_location)
    run_id = run_ids[0]
    data = load_by_id(run_id)
    assert data.parameters == 'ch1,ch2,voltage'
    assert data.number_of_results == 36
    expected_names = ['ch1', 'ch2', 'voltage']
    expected_labels = ['Gate ch1', 'Gate ch2', 'Gate voltage']
    expected_units = ['V', 'V', 'V']
    expected_depends_on = ['', '', 'ch1, ch2']
    for i, parameter in enumerate(data.get_parameters()):
        assert parameter.name == expected_names[i]
        assert parameter.label == expected_labels[i]
        assert parameter.unit == expected_units[i]
        assert parameter.depends_on == expected_depends_on[i]
        assert parameter.type == 'numeric'
    snapshot = json.loads(data.get_metadata('snapshot'))
    assert sorted(list(snapshot.keys())) == ['__class__', 'arrays',
                                             'formatter', 'io', 'location',
                                             'loop', 'station']


def test_load_legacy_files_1D(experiment):
    location = 'fixtures/2018-01-17/#001_testsweep_15-42-57'
    dir = os.path.dirname(__file__)
    full_location = os.path.join(dir, location)
    run_ids = import_dat_file(full_location)
    run_id = run_ids[0]
    data = load_by_id(run_id)
    assert data.parameters == 'ch1,voltage'
    assert data.number_of_results == 201
    expected_names = ['ch1', 'voltage']
    expected_labels = ['Gate ch1', 'Gate voltage']
    expected_units = ['V', 'V']
    expected_depends_on = ['', 'ch1']
    for i, parameter in enumerate(data.get_parameters()):
        assert parameter.name == expected_names[i]
        assert parameter.label == expected_labels[i]
        assert parameter.unit == expected_units[i]
        assert parameter.depends_on == expected_depends_on[i]
        assert parameter.type == 'numeric'
    snapshot = json.loads(data.get_metadata('snapshot'))
    assert sorted(list(snapshot.keys())) == ['__class__', 'arrays',
                                             'formatter', 'io', 'location',
                                             'loop', 'station']
