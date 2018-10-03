import pytest
import tempfile
import os
from time import sleep
import json

from hypothesis import given, settings
import hypothesis.strategies as hst
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from qcodes.tests.common import retry_until_does_not_throw

import qcodes as qc
from qcodes.dataset.data_export import get_data_by_id
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.experiment_container import new_experiment
from qcodes.tests.instrument_mocks import DummyInstrument, \
    DummyChannelInstrument, setpoint_generator
from qcodes.dataset.param_spec import ParamSpec
from qcodes.dataset.sqlite_base import atomic_transaction
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
def channel_array_instrument():
    channelarrayinstrument = DummyChannelInstrument('dummy_channel_inst')
    yield channelarrayinstrument
    channelarrayinstrument.close()


@pytest.fixture
def SpectrumAnalyzer():
    """
    Yields a DummyInstrument that holds ArrayParameters returning
    different types
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

    class MultiDimSpectrum(ArrayParameter):

        def __init__(self, name, instrument):
            self.start = 0
            self.stop = 2e6
            self.npts = (100, 50, 20)
            sp1 = np.linspace(self.start, self.stop,
                              self.npts[0])
            sp2 = np.linspace(self.start, self.stop,
                              self.npts[1])
            sp3 = np.linspace(self.start, self.stop,
                              self.npts[2])
            setpoints = setpoint_generator(sp1, sp2, sp3)

            super().__init__(name=name,
                             instrument=instrument,
                             setpoints=setpoints,
                             shape=(100, 50, 20),
                             label='Flower Power Spectrum in 3D',
                             unit='V/sqrt(Hz)',
                             setpoint_names=('Frequency0', 'Frequency1',
                                             'Frequency2'),
                             setpoint_units=('Hz', 'Other Hz', "Third Hz"))

        def get_raw(self):
            return np.random.randn(*self.npts)

    class ListSpectrum(Spectrum):

        def get_raw(self):
            output = super().get_raw()
            return list(output)

    class TupleSpectrum(Spectrum):

        def get_raw(self):
            output = super().get_raw()
            return tuple(output)

    SA = DummyInstrument('dummy_SA')
    SA.add_parameter('spectrum', parameter_class=Spectrum)
    SA.add_parameter('listspectrum', parameter_class=ListSpectrum)
    SA.add_parameter('tuplespectrum', parameter_class=TupleSpectrum)
    SA.add_parameter('multidimspectrum', parameter_class=MultiDimSpectrum)
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


def test_adding_scalars_as_array_raises(experiment, DAC):
    """
    Test that adding scalars to an array type parameter raises
    """
    meas = Measurement()
    meas.register_parameter(DAC.ch1, paramtype='array')
    meas.register_parameter(DAC.ch2, paramtype='array')

    with meas.run() as datasaver:
        with pytest.raises(ValueError):
            datasaver.add_result((DAC.ch1, DAC.ch1()),
                                 (DAC.ch2, DAC.ch2()))


def test_mixing_array_and_numeric_raises(experiment, DAC):
    """
    Test that mixing array and numeric types raises
    """
    meas = Measurement()
    meas.register_parameter(DAC.ch1, paramtype='numeric')
    meas.register_parameter(DAC.ch2, paramtype='array')

    with meas.run() as datasaver:
        with pytest.raises(RuntimeError):
            datasaver.add_result((DAC.ch1, np.array([DAC.ch1(), DAC.ch1()])),
                                 (DAC.ch2, np.array([DAC.ch2(), DAC.ch1()])))

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


@settings(deadline=None)
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
        assert meas._write_period == float(wp)

        with meas.run() as datasaver:
            assert datasaver.write_period == float(wp)


@settings(deadline=None)
@given(words=hst.lists(elements=hst.text(), min_size=4, max_size=10))
def test_enter_and_exit_actions(experiment, DAC, words):
    # we use a list to check that the functions executed
    # in the correct order

    def action(lst, word):
        lst.append(word)

    meas = Measurement()
    meas.register_parameter(DAC.ch1)

    testlist = []

    splitpoint = round(len(words) / 2)
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
    """
    Test that subscribers are called at the moment the data is flushed to database

    Note that for the purpose of this test, flush_data_to_database method is
    called explicitly instead of waiting for the data to be flushed
    automatically after the write_period passes after a add_result call.
    """

    def collect_all_results(results, length, state):
        """
        Updates the *state* to contain all the *results* acquired
        during the experiment run
        """
        # Due to the fact that by default subscribers only hold 1 data value
        # in their internal queue, this assignment should work (i.e. not
        # overwrite values in the "state" object) assuming that at the start
        # of the experiment both the dataset and the *state* objects have
        # the same length.
        state[length] = results

    def collect_values_larger_than_7(results, length, state):
        """
        Appends to the *state* only the values from *results*
        that are larger than 7
        """
        for result_tuple in results:
            state += [value for value in result_tuple if value > 7]

    meas = Measurement(exp=experiment)
    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DMM.v1, setpoints=(DAC.ch1,))

    # key is the number of the result tuple,
    # value is the result tuple itself
    all_results_dict = {}
    values_larger_than_7 = []

    meas.add_subscriber(collect_all_results, state=all_results_dict)
    assert len(meas.subscribers) == 1
    meas.add_subscriber(collect_values_larger_than_7,
                        state=values_larger_than_7)
    assert len(meas.subscribers) == 2

    meas.write_period = 0.2

    with meas.run() as datasaver:

        # Assert that the measurement, runner, and datasaver
        # have added subscribers to the dataset
        assert len(datasaver._dataset.subscribers) == 2

        assert all_results_dict == {}
        assert values_larger_than_7 == []

        dac_vals_and_dmm_vals = list(zip(range(5), range(3, 8)))
        values_larger_than_7__expected = []

        for num in range(5):
            (dac_val, dmm_val) = dac_vals_and_dmm_vals[num]
            values_larger_than_7__expected += \
                [val for val in (dac_val, dmm_val) if val > 7]

            datasaver.add_result((DAC.ch1, dac_val), (DMM.v1, dmm_val))

            # Ensure that data is flushed to the database despite the write
            # period, so that the database triggers are executed, which in turn
            # add data to the queues within the subscribers
            datasaver.flush_data_to_database()

            # In order to make this test deterministic, we need to ensure that
            # just enough time has passed between the moment the data is flushed
            # to database and the "state" object (that is passed to subscriber
            # constructor) has been updated by the corresponding subscriber's
            # callback function. At the moment, there is no robust way to ensure
            # this. The reason is that the subscribers have internal queue which
            # is populated via a trigger call from the SQL database, hence from
            # this "main" thread it is difficult to say whether the queue is
            # empty because the subscriber callbacks have already been executed
            # or because the triggers of the SQL database has not been executed
            # yet.
            #
            # In order to overcome this problem, a special decorator is used
            # to wrap the assertions. This is going to ensure that some time
            # is given to the Subscriber threads to finish exhausting the queue.
            @retry_until_does_not_throw(
                exception_class_to_expect=AssertionError, delay=0.5, tries=10)
            def assert_states_updated_from_callbacks():
                assert values_larger_than_7 == values_larger_than_7__expected
                assert list(all_results_dict.keys()) == \
                       [result_index for result_index in range(1, num + 1 + 1)]

            assert_states_updated_from_callbacks()

    # Ensure that after exiting the "run()" context,
    # all subscribers get unsubscribed from the dataset
    assert len(datasaver._dataset.subscribers) == 0

    # Ensure that the triggers for each subscriber
    # have been removed from the database
    get_triggers_sql = "SELECT * FROM sqlite_master WHERE TYPE = 'trigger';"
    triggers = atomic_transaction(
        datasaver._dataset.conn, get_triggers_sql).fetchall()
    assert len(triggers) == 0


def test_subscribers_called_at_exiting_context_if_queue_is_not_empty(experiment,
                                                                     DAC):
    """
    Upon quitting the "run()" context, verify that in case the queue is
    not empty, the subscriber's callback is still called on that data.
    This situation is created by setting the minimum length of the queue
    to a number that is larger than the number of value written to the dataset.
    """

    def collect_x_vals(results, length, state):
        """
        Collects first elements of results tuples in *state*
        """
        index_of_x = 0
        state += [res[index_of_x] for res in results]

    meas = Measurement(exp=experiment)
    meas.register_parameter(DAC.ch1)

    collected_x_vals = []

    meas.add_subscriber(collect_x_vals, state=collected_x_vals)

    given_x_vals = [0, 1, 2, 3]

    with meas.run() as datasaver:
        # Set the minimum queue size of the subscriber to more that
        # the total number of values being added to the dataset;
        # this way the subscriber callback is not called before
        # we exit the "run()" context.
        subscriber = list(datasaver.dataset.subscribers.values())[0]
        subscriber.min_queue_length = int(len(given_x_vals) + 1)

        for x in given_x_vals:
            datasaver.add_result((DAC.ch1, x))
            # Verify that the subscriber callback is not called yet
            assert collected_x_vals == []

    # Verify that the subscriber callback is finally called
    assert collected_x_vals == given_x_vals


@settings(deadline=None, max_examples=25)
@given(N=hst.integers(min_value=2000, max_value=3000))
def test_subscribers_called_for_all_data_points(experiment, DAC, DMM, N):
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
    given_yvals = range(1, N + 1)

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
        for set_v, get_v in zip(set_values[:breakpoint],
                                get_values[:breakpoint]):
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
def test_datasaver_arrays_lists_tuples(empty_temp_db, N):
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
        signal = np.random.randn(N - 1)

        with pytest.raises(ValueError):
            datasaver.add_result(('freqax', freqax), ('signal', signal))

    meas.register_custom_parameter(name='gate_voltage',
                                   label='Gate tuning potential',
                                   unit='V')
    meas.register_custom_parameter(name='signal',
                                   label='qubit signal',
                                   unit='Majorana flux',
                                   setpoints=('freqax', 'gate_voltage'))

    # save arrays
    with meas.run() as datasaver:
        freqax = np.linspace(1e6, 2e6, N)
        signal = np.random.randn(N)

        datasaver.add_result(('freqax', freqax),
                             ('signal', signal),
                             ('gate_voltage', 0))

    assert datasaver.points_written == N

    # save lists
    with meas.run() as datasaver:
        freqax = list(np.linspace(1e6, 2e6, N))
        signal = list(np.random.randn(N))

        datasaver.add_result(('freqax', freqax),
                             ('signal', signal),
                             ('gate_voltage', 0))

    assert datasaver.points_written == N

    # save tuples
    with meas.run() as datasaver:
        freqax = tuple(np.linspace(1e6, 2e6, N))
        signal = tuple(np.random.randn(N))

        datasaver.add_result(('freqax', freqax),
                             ('signal', signal),
                             ('gate_voltage', 0))

    assert datasaver.points_written == N


def test_datasaver_foul_input(experiment):
    meas = Measurement()

    meas.register_custom_parameter('foul',
                                   label='something unnatural',
                                   unit='Fahrenheit')

    foul_stuff = [qc.Parameter('foul'), set((1, 2, 3))]

    with meas.run() as datasaver:
        for ft in foul_stuff:
            with pytest.raises(ValueError):
                datasaver.add_result(('foul', ft))


@settings(max_examples=10, deadline=None)
@given(N=hst.integers(min_value=2, max_value=500))
def test_datasaver_unsized_arrays(empty_temp_db, N):
    new_experiment('firstexp', sample_name='no sample')

    meas = Measurement()

    meas.register_custom_parameter(name='freqax',
                                   label='Frequency axis',
                                   unit='Hz')
    meas.register_custom_parameter(name='signal',
                                   label='qubit signal',
                                   unit='Majorana number',
                                   setpoints=('freqax',))
    # note that np.array(some_number) is not the same as the number
    # its also not an array with a shape. Check here that we handle it
    # correctly
    with meas.run() as datasaver:
        freqax = np.linspace(1e6, 2e6, N)
        signal = np.random.randn(N)
        for i in range(N):
            myfreq = np.array(freqax[i])
            mysignal = np.array(signal[i])
            datasaver.add_result(('freqax', myfreq), ('signal', mysignal))

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

    assert datasaver.points_written == N * M


@settings(max_examples=5, deadline=None)
@given(N=hst.integers(min_value=5, max_value=500),
       M=hst.integers(min_value=4, max_value=250))
def test_datasaver_arrayparams_lists(experiment, SpectrumAnalyzer, DAC, N, M):
    lspec = SpectrumAnalyzer.listspectrum

    meas = Measurement()

    meas.register_parameter(lspec)
    assert len(meas.parameters) == 2
    assert meas.parameters[str(lspec)].depends_on == 'dummy_SA_Frequency'
    assert meas.parameters[str(lspec)].type == 'numeric'
    assert meas.parameters['dummy_SA_Frequency'].type == 'numeric'

    # Now for a real measurement

    meas = Measurement()

    meas.register_parameter(DAC.ch1)
    meas.register_parameter(lspec, setpoints=[DAC.ch1])

    assert len(meas.parameters) == 3

    lspec.npts = M

    with meas.run() as datasaver:
        for set_v in np.linspace(0, 0.01, N):
            datasaver.add_result((DAC.ch1, set_v),
                                 (lspec, lspec.get()))

    assert datasaver.points_written == N * M


@settings(max_examples=5, deadline=None)
@given(N=hst.integers(min_value=5, max_value=500),
       M=hst.integers(min_value=4, max_value=250))
def test_datasaver_arrayparams_tuples(experiment, SpectrumAnalyzer, DAC, N, M):
    tspec = SpectrumAnalyzer.tuplespectrum

    meas = Measurement()

    meas.register_parameter(tspec)
    assert len(meas.parameters) == 2
    assert meas.parameters[str(tspec)].depends_on == 'dummy_SA_Frequency'
    assert meas.parameters[str(tspec)].type == 'numeric'
    assert meas.parameters['dummy_SA_Frequency'].type == 'numeric'

    # Now for a real measurement

    meas = Measurement()

    meas.register_parameter(DAC.ch1)
    meas.register_parameter(tspec, setpoints=[DAC.ch1])

    assert len(meas.parameters) == 3

    tspec.npts = M

    with meas.run() as datasaver:
        for set_v in np.linspace(0, 0.01, N):
            datasaver.add_result((DAC.ch1, set_v),
                                 (tspec, tspec.get()))

    assert datasaver.points_written == N * M


@settings(max_examples=5, deadline=None)
@given(N=hst.integers(min_value=5, max_value=500))
def test_datasaver_array_parameters_channel(experiment,
                                            channel_array_instrument,
                                            DAC, N):
    meas = Measurement()

    array_param = channel_array_instrument.A.dummy_array_parameter

    meas.register_parameter(array_param)

    assert len(meas.parameters) == 2
    dependency_name = 'dummy_channel_inst_ChanA_this_setpoint'
    assert meas.parameters[str(array_param)].depends_on == dependency_name
    assert meas.parameters[str(array_param)].type == 'numeric'
    assert meas.parameters[dependency_name].type == 'numeric'

    # Now for a real measurement

    meas = Measurement()

    meas.register_parameter(DAC.ch1)
    meas.register_parameter(array_param, setpoints=[DAC.ch1])

    assert len(meas.parameters) == 3

    M = array_param.shape[0]

    with meas.run() as datasaver:
        for set_v in np.linspace(0, 0.01, N):
            datasaver.add_result((DAC.ch1, set_v),
                                 (array_param, array_param.get()))
    assert datasaver.points_written == N * M

    expected_params = ('dummy_dac_ch1',
                       'dummy_channel_inst_ChanA_this_setpoint',
                       'dummy_channel_inst_ChanA_dummy_array_parameter')
    ds = load_by_id(datasaver.run_id)
    for param in expected_params:
        data = ds.get_data(param)
        assert len(data) == N * M
        assert len(data[0]) == 1
    datadicts = get_data_by_id(datasaver.run_id)
    # one dependent parameter
    assert len(datadicts) == 1
    datadicts = datadicts[0]
    assert len(datadicts) == len(meas.parameters)
    for datadict in datadicts:
        assert datadict['data'].shape == (N * M,)


@settings(max_examples=5, deadline=None)
@given(N=hst.integers(min_value=5, max_value=500))
def test_datasaver_array_parameters_array(experiment, channel_array_instrument,
                                          DAC, N):
    """
    Test that storing array parameters inside a loop with the sqlite
    Array type works as expected
    """
    meas = Measurement()

    array_param = channel_array_instrument.A.dummy_array_parameter

    meas.register_parameter(array_param, paramtype='array')

    assert len(meas.parameters) == 2
    dependency_name = 'dummy_channel_inst_ChanA_this_setpoint'
    assert meas.parameters[str(array_param)].depends_on == dependency_name
    assert meas.parameters[str(array_param)].type == 'array'
    assert meas.parameters[dependency_name].type == 'array'

    # Now for a real measurement

    meas = Measurement()

    meas.register_parameter(DAC.ch1, paramtype='numeric')
    meas.register_parameter(array_param, setpoints=[DAC.ch1], paramtype='array')

    assert len(meas.parameters) == 3

    M = array_param.shape[0]
    dac_datapoints = np.linspace(0, 0.01, N)
    with meas.run() as datasaver:
        for set_v in dac_datapoints:
            datasaver.add_result((DAC.ch1, set_v),
                                 (array_param, array_param.get()))
    assert datasaver.points_written == N
    ds = load_by_id(datasaver.run_id)

    data_num = ds.get_data('dummy_dac_ch1')
    assert len(data_num) == N

    setpoint_arrays = ds.get_data('dummy_channel_inst_ChanA_this_setpoint')
    data_arrays = ds.get_data('dummy_channel_inst_ChanA_dummy_array_parameter')
    assert len(setpoint_arrays) == N
    assert len(data_arrays) == N

    for data_arrays, setpoint_array in zip(data_arrays, setpoint_arrays):
        assert_array_equal(setpoint_array[0], np.linspace(5, 9, 5))
        assert_array_equal(data_arrays[0], np.array([2., 2., 2., 2., 2.]))

    datadicts = get_data_by_id(datasaver.run_id)
    # one dependent parameter
    assert len(datadicts) == 1
    datadicts = datadicts[0]
    assert len(datadicts) == len(meas.parameters)
    for datadict in datadicts:
        if datadict['name'] == 'dummy_dac_ch1':
            expected_data = np.repeat(dac_datapoints, M)
        if datadict['name'] == 'dummy_channel_inst_ChanA_this_setpoint':
            expected_data = np.tile(np.linspace(5, 9, 5), N)
        if datadict['name'] == 'dummy_channel_inst_ChanA_dummy_array_parameter':
            expected_data = np.empty(N * M)
            expected_data[:] = 2.
        assert_allclose(datadict['data'], expected_data)

        assert datadict['data'].shape == (N * M,)


def test_datasaver_multidim_array(experiment):
    """
    Test that inserting multidim parameters as arrays works as expected
    """
    meas = Measurement(experiment)
    size1 = 10
    size2 = 15

    data_mapping = {name: i for i, name in
                    zip(range(4), ['x1', 'x2', 'y1', 'y2'])}

    x1 = qc.ManualParameter('x1')
    x2 = qc.ManualParameter('x2')
    y1 = qc.ManualParameter('y1')
    y2 = qc.ManualParameter('y2')

    meas.register_parameter(x1, paramtype='array')
    meas.register_parameter(x2, paramtype='array')
    meas.register_parameter(y1, setpoints=[x1, x2], paramtype='array')
    meas.register_parameter(y2, setpoints=[x1, x2], paramtype='array')
    data = np.random.rand(4, size1, size2)
    with meas.run() as datasaver:
        datasaver.add_result((str(x1), data[0, :, :]),
                             (str(x2), data[1, :, :]),
                             (str(y1), data[2, :, :]),
                             (str(y2), data[3, :, :]))
    assert datasaver.points_written == 1
    dataset = load_by_id(datasaver.run_id)
    for myid, expected in zip(('x1', 'x2', 'y1', 'y2'), data):
        mydata = dataset.get_data(myid)
        assert len(mydata) == 1
        assert len(mydata[0]) == 1
        assert mydata[0][0].shape == (size1, size2)
        assert_array_equal(mydata[0][0], expected)

    datadicts = get_data_by_id(datasaver.run_id)
    assert len(datadicts) == 2
    for datadict_list in datadicts:
        assert len(datadict_list) == 3
        for datadict in datadict_list:
            dataindex = data_mapping[datadict['name']]
            expected_data = data[dataindex, :, :].ravel()
            assert_allclose(datadict['data'], expected_data)

            assert datadict['data'].shape == (size1 * size2,)


def test_datasaver_multidim_numeric(experiment):
    """
    Test that inserting multidim parameters as numeric works as expected
    """
    meas = Measurement(experiment)
    size1 = 10
    size2 = 15
    x1 = qc.ManualParameter('x1')
    x2 = qc.ManualParameter('x2')
    y1 = qc.ManualParameter('y1')
    y2 = qc.ManualParameter('y2')

    data_mapping = {name: i for i, name in
                    zip(range(4), ['x1', 'x2', 'y1', 'y2'])}

    meas.register_parameter(x1, paramtype='numeric')
    meas.register_parameter(x2, paramtype='numeric')
    meas.register_parameter(y1, setpoints=[x1, x2], paramtype='numeric')
    meas.register_parameter(y2, setpoints=[x1, x2], paramtype='numeric')
    data = np.random.rand(4, size1, size2)
    with meas.run() as datasaver:
        datasaver.add_result((str(x1), data[0, :, :]),
                             (str(x2), data[1, :, :]),
                             (str(y1), data[2, :, :]),
                             (str(y2), data[3, :, :]))
    assert datasaver.points_written == size1 * size2
    dataset = load_by_id(datasaver.run_id)
    for myid, expected in zip(('x1', 'x2', 'y1', 'y2'), data):
        mydata = dataset.get_data(myid)
        assert len(mydata) == size1 * size2
        assert len(mydata[0]) == 1
        assert isinstance(mydata[0][0], float)
        assert_allclose(np.array(mydata).ravel(), expected.ravel())

    datadicts = get_data_by_id(datasaver.run_id)
    assert len(datadicts) == 2
    for datadict_list in datadicts:
        assert len(datadict_list) == 3
        for datadict in datadict_list:
            dataindex = data_mapping[datadict['name']]
            expected_data = data[dataindex, :, :].ravel()
            assert_allclose(datadict['data'], expected_data)

            assert datadict['data'].shape == (size1 * size2,)


def test_datasaver_multidimarrayparameter_as_array(experiment,
                                                   SpectrumAnalyzer):
    """
    Test that inserting multidim Arrrayparameters as array works as expected
    """
    array_param = SpectrumAnalyzer.multidimspectrum
    meas = Measurement()
    meas.register_parameter(array_param, paramtype='array')
    assert len(meas.parameters) == 4
    inserted_data = array_param.get()
    with meas.run() as datasaver:
        datasaver.add_result((array_param, inserted_data))

    assert datasaver.points_written == 1
    ds = load_by_id(datasaver.run_id)
    expected_shape = (100, 50, 20)
    for i in range(3):
        data = ds.get_data(f'dummy_SA_Frequency{i}')[0][0]
        aux_shape = list(expected_shape)
        aux_shape.pop(i)

        assert data.shape == expected_shape
        for j in range(aux_shape[0]):
            for k in range(aux_shape[1]):
                # todo There should be a simpler way of doing this
                if i == 0:
                    mydata = data[:, j, k]
                if i == 1:
                    mydata = data[j, :, k]
                if i == 2:
                    mydata = data[j, k, :]
                assert_array_equal(mydata,
                                   np.linspace(array_param.start,
                                               array_param.stop,
                                               array_param.npts[i]))

    datadicts = get_data_by_id(datasaver.run_id)
    assert len(datadicts) == 1
    for datadict_list in datadicts:
        assert len(datadict_list) == 4
        for i, datadict in enumerate(datadict_list):

            datadict['data'].shape = (np.prod(expected_shape),)
            if i == 0:
                temp_data = np.linspace(array_param.start,
                                        array_param.stop,
                                        array_param.npts[0])
                expected_data = np.repeat(temp_data,
                                          expected_shape[1] * expected_shape[2])
            if i == 1:
                temp_data = np.linspace(array_param.start,
                                        array_param.stop,
                                        array_param.npts[i])
                expected_data = np.tile(np.repeat(temp_data, expected_shape[2]),
                                        expected_shape[0])
            if i == 2:
                temp_data = np.linspace(array_param.start,
                                        array_param.stop,
                                        array_param.npts[i])
                expected_data = np.tile(temp_data,
                                        expected_shape[0] * expected_shape[1])
            if i == 3:
                expected_data = inserted_data.ravel()
            assert_allclose(datadict['data'], expected_data)


def test_datasaver_multidimarrayparameter_as_numeric(experiment,
                                                     SpectrumAnalyzer):
    """
    Test that storing a multidim Array parameter as numeric unravels the
    parameter as expected.
    """

    array_param = SpectrumAnalyzer.multidimspectrum
    meas = Measurement()
    meas.register_parameter(array_param, paramtype='numeric')
    expected_shape = array_param.shape
    dims = len(array_param.shape)
    assert len(meas.parameters) == dims + 1

    points_expected = np.prod(array_param.npts)
    inserted_data = array_param.get()
    with meas.run() as datasaver:
        datasaver.add_result((array_param, inserted_data))

    assert datasaver.points_written == points_expected
    ds = load_by_id(datasaver.run_id)
    # check setpoints

    expected_setpoints_vectors = (np.linspace(array_param.start,
                                              array_param.stop,
                                              array_param.npts[i]) for i in
                                  range(dims))
    expected_setpoints_matrix = np.meshgrid(*expected_setpoints_vectors,
                                            indexing='ij')
    expected_setpoints = tuple(
        setpoint_array.ravel() for setpoint_array in expected_setpoints_matrix)

    for i in range(dims):
        data = ds.get_data(f'dummy_SA_Frequency{i}')
        assert len(data) == points_expected
        assert_allclose(np.array(data).squeeze(),
                        expected_setpoints[i])
    data = np.array(ds.get_data('dummy_SA_multidimspectrum')).squeeze()
    assert_allclose(data, inserted_data.ravel())

    datadicts = get_data_by_id(datasaver.run_id)
    assert len(datadicts) == 1
    for datadict_list in datadicts:
        assert len(datadict_list) == 4
        for i, datadict in enumerate(datadict_list):

            datadict['data'].shape = (np.prod(expected_shape),)
            if i == 0:
                temp_data = np.linspace(array_param.start,
                                        array_param.stop,
                                        array_param.npts[0])
                expected_data = np.repeat(temp_data,
                                          expected_shape[1] * expected_shape[2])
            if i == 1:
                temp_data = np.linspace(array_param.start,
                                        array_param.stop,
                                        array_param.npts[i])
                expected_data = np.tile(np.repeat(temp_data, expected_shape[2]),
                                        expected_shape[0])
            if i == 2:
                temp_data = np.linspace(array_param.start,
                                        array_param.stop,
                                        array_param.npts[i])
                expected_data = np.tile(temp_data,
                                        expected_shape[0] * expected_shape[1])
            if i == 3:
                expected_data = inserted_data.ravel()
            assert_allclose(datadict['data'], expected_data)


def test_datasaver_multi_parameters_scalar(experiment,
                                           channel_array_instrument):
    """
    Test that we can register multiparameters that are scalar.
    """
    meas = Measurement()
    param = channel_array_instrument.A.dummy_scalar_multi_parameter
    meas.register_parameter(param)
    assert len(meas.parameters) == len(param.shapes)
    assert tuple(meas.parameters.keys()) == tuple(param.names)

    with meas.run() as datasaver:
        datasaver.add_result((param, param()))

    assert datasaver.points_written == 1
    ds = load_by_id(datasaver.run_id)
    assert ds.get_data('thisparam') == [[0]]
    assert ds.get_data('thatparam') == [[1]]


def test_datasaver_multi_parameters_array(experiment,
                                          channel_array_instrument):
    """
    Test that we can register multiparameters that are array like.
    """
    meas = Measurement()
    param = channel_array_instrument.A.dummy_multi_parameter
    meas.register_parameter(param)
    assert len(meas.parameters) == 3  # two params + 1D identical setpoints
    param_names = ('dummy_channel_inst_ChanA_this_setpoint',
                   'this', 'that')
    assert tuple(meas.parameters.keys()) == param_names
    assert meas.parameters[
               'this'].depends_on == 'dummy_channel_inst_ChanA_this_setpoint'
    assert meas.parameters[
               'that'].depends_on == 'dummy_channel_inst_ChanA_this_setpoint'
    assert meas.parameters[
               'dummy_channel_inst_ChanA_this_setpoint'].depends_on == ''

    with meas.run() as datasaver:
        datasaver.add_result((param, param()))
    assert datasaver.points_written == 5
    ds = load_by_id(datasaver.run_id)
    assert ds.get_data('dummy_channel_inst_ChanA_this_setpoint') == [[5],
                                                                     [6],
                                                                     [7],
                                                                     [8],
                                                                     [9]]
    assert ds.get_data('this') == [[0], [0], [0], [0], [0]]
    assert ds.get_data('that') == [[1], [1], [1], [1], [1]]


def test_datasaver_2d_multi_parameters_array(experiment,
                                             channel_array_instrument):
    """
    Test that we can register multiparameters that are array like and 2D.
    """
    meas = Measurement()
    param = channel_array_instrument.A.dummy_2d_multi_parameter
    meas.register_parameter(param)
    assert len(meas.parameters) == 4  # two params + 2D identical setpoints
    param_names = ('dummy_channel_inst_ChanA_this_setpoint',
                   'dummy_channel_inst_ChanA_that_setpoint',
                   'this', 'that')
    assert tuple(meas.parameters.keys()) == param_names
    assert meas.parameters[
               'this'].depends_on == 'dummy_channel_inst_ChanA_this_setpoint' \
                                     ', dummy_channel_inst_ChanA_that_setpoint'
    assert meas.parameters[
               'that'].depends_on == 'dummy_channel_inst_ChanA_this_setpoint' \
                                     ', dummy_channel_inst_ChanA_that_setpoint'
    assert meas.parameters[
               'dummy_channel_inst_ChanA_this_setpoint'].depends_on == ''
    assert meas.parameters[
               'dummy_channel_inst_ChanA_that_setpoint'].depends_on == ''

    with meas.run() as datasaver:
        datasaver.add_result((param, param()))

    assert datasaver.points_written == 15
    ds = load_by_id(datasaver.run_id)

    assert ds.get_data('dummy_channel_inst_ChanA_this_setpoint') == [[5],
                                                                     [5],
                                                                     [5],
                                                                     [6],
                                                                     [6],
                                                                     [6],
                                                                     [7],
                                                                     [7],
                                                                     [7],
                                                                     [8],
                                                                     [8],
                                                                     [8],
                                                                     [9],
                                                                     [9],
                                                                     [9]]
    assert ds.get_data('dummy_channel_inst_ChanA_that_setpoint') == [[9],
                                                                     [10],
                                                                     [11],
                                                                     [9],
                                                                     [10],
                                                                     [11],
                                                                     [9],
                                                                     [10],
                                                                     [11],
                                                                     [9],
                                                                     [10],
                                                                     [11],
                                                                     [9],
                                                                     [10],
                                                                     [11]]

    assert ds.get_data('this') == [[0], [0], [0], [0], [0],
                                   [0], [0], [0], [0], [0],
                                   [0], [0], [0], [0], [0]]
    assert ds.get_data('that') == [[1], [1], [1], [1], [1],
                                   [1], [1], [1], [1], [1],
                                   [1], [1], [1], [1], [1]]


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
