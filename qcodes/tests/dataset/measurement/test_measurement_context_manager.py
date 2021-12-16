import json
import logging
import os
import re
import traceback
from time import sleep

import hypothesis.strategies as hst
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from numpy.testing import assert_allclose, assert_array_equal

import qcodes as qc
from qcodes.dataset.data_set import DataSet, load_by_id
from qcodes.dataset.descriptions.param_spec import ParamSpecBase
from qcodes.dataset.experiment_container import new_experiment
from qcodes.dataset.export_config import DataExportType
from qcodes.dataset.legacy_import import import_dat_file
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.sqlite.connection import atomic_transaction
from qcodes.instrument.parameter import (
    ArrayParameter,
    Parameter,
    expand_setpoints_helper,
)
from qcodes.tests.common import reset_config_on_exit, retry_until_does_not_throw

# pylint: disable=unused-import
from qcodes.tests.test_station import set_default_station_to_none
from qcodes.utils.validators import Arrays


def test_log_messages(caplog, meas_with_registered_param):
    caplog.set_level(logging.INFO)

    with meas_with_registered_param.run():
        pass

    assert "Set the run_timestamp of run_id" in caplog.text
    assert "Starting measurement with guid" in caplog.text
    assert "Finished measurement with guid" in caplog.text


def test_log_includes_extra_info(caplog, meas_with_registered_param):
    caplog.set_level(logging.INFO)
    meas_with_registered_param._extra_log_info = "some extra info"
    with meas_with_registered_param.run():
        pass

    assert "some extra info" in caplog.text


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

    # we allow the registration of the EXACT same parameter twice...
    meas.register_parameter(my_param)
    # ... but not a different parameter with a new name
    attrs = ['label', 'unit']
    vals = ['new label', 'new unit']
    for attr, val in zip(attrs, vals):
        old_val = getattr(my_param, attr)
        setattr(my_param, attr, val)
        match = re.escape("Parameter already registered in this Measurement.")
        with pytest.raises(ValueError, match=match):
            meas.register_parameter(my_param)
        setattr(my_param, attr, old_val)

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
    meas.unregister_parameter(my_param)
    meas.register_parameter(my_param, basis=(DAC.ch2,),
                            setpoints=(DMM.v1, DMM.v2))

    assert set(meas.parameters.keys()) == {str(DAC.ch2),
                                               str(DMM.v1), str(DMM.v2),
                                               str(my_param)}
    paramspec = meas.parameters[str(my_param)]
    assert paramspec.name == str(my_param)

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
    assert isinstance(meas.parameters[name], ParamSpecBase)
    assert meas.parameters[name].unit == unit
    assert meas.parameters[name].label == label
    assert meas.parameters[name].type == 'numeric'

    newunit = 'V^3'
    newlabel = 'cube of the voltage'

    meas.unregister_parameter(name)
    meas.register_custom_parameter(name, newlabel, newunit)

    assert len(meas.parameters) == 1
    assert isinstance(meas.parameters[name], ParamSpecBase)
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

    meas.unregister_parameter(name)
    meas.register_custom_parameter(name, label, unit,
                                   setpoints=(DAC.ch1, str(DAC.ch2)),
                                   basis=('strange_dac',))

    assert len(meas.parameters) == 4
    parspec = meas.parameters[name]

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
    assert set(meas.parameters.keys()) == {str(DAC.ch2), str(DMM.v1),
                                               str(DMM.v2)}

    meas.unregister_parameter(DAC.ch2)
    assert set(meas.parameters.keys()) == {str(DMM.v1), str(DMM.v2)}

    not_parameters = [DAC, DMM, 0.0, 1]
    for notparam in not_parameters:
        with pytest.raises(ValueError):
            meas.unregister_parameter(notparam)

    # unregistering something not registered should silently "succeed"
    meas.unregister_parameter('totes_not_registered')
    meas.unregister_parameter(DAC.ch2)
    meas.unregister_parameter(DAC.ch2)


@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize("bg_writing", [True, False])
def test_mixing_array_and_numeric(DAC, bg_writing):
    """
    Test that mixing array and numeric types is okay
    """
    meas = Measurement()
    meas.register_parameter(DAC.ch1, paramtype='numeric')
    meas.register_parameter(DAC.ch2, paramtype='array')

    with meas.run(write_in_background=bg_writing) as datasaver:
        datasaver.add_result((DAC.ch1, np.array([DAC.ch1(), DAC.ch1()])),
                             (DAC.ch2, np.array([DAC.ch2(), DAC.ch1()])))


def test_measurement_name_default(experiment, DAC, DMM):
    fmt = experiment.format_string
    exp_id = experiment.exp_id

    default_name = 'results'

    meas = Measurement()
    assert meas.name == ''

    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DMM.v1, setpoints=[DAC.ch1])

    with meas.run() as datasaver:
        run_id = datasaver.run_id
        expected_name = fmt.format(default_name, exp_id, run_id)
        ds = datasaver.dataset
        assert isinstance(ds, DataSet)
        assert ds.table_name == expected_name
        assert ds.name == default_name


def test_measurement_name_changed_via_attribute(experiment, DAC, DMM):
    fmt = experiment.format_string
    exp_id = experiment.exp_id

    name = 'yolo'

    meas = Measurement()
    meas.name = name

    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DMM.v1, setpoints=[DAC.ch1])

    with meas.run() as datasaver:
        run_id = datasaver.run_id
        expected_name = fmt.format('results', exp_id, run_id)
        ds = datasaver.dataset
        assert isinstance(ds, DataSet)
        assert ds.table_name == expected_name
        assert ds.name == name


def test_measurement_name_set_as_argument(experiment, DAC, DMM):
    fmt = experiment.format_string
    exp_id = experiment.exp_id

    name = 'yolo'

    meas = Measurement(name=name, exp=experiment)

    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DMM.v1, setpoints=[DAC.ch1])

    with meas.run() as datasaver:
        run_id = datasaver.run_id
        expected_name = fmt.format('results', exp_id, run_id)
        ds = datasaver.dataset
        assert isinstance(ds, DataSet)
        assert ds.table_name == expected_name
        assert ds.name == name


@settings(deadline=None)
@given(wp=hst.one_of(hst.integers(), hst.floats(allow_nan=False),
                     hst.text()))
@pytest.mark.usefixtures("empty_temp_db")
def test_setting_write_period(wp):
    new_experiment('firstexp', sample_name='no sample')
    meas = Measurement()
    meas.register_custom_parameter(name='dummy')

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
@given(wp=hst.one_of(hst.integers(), hst.floats(allow_nan=False),
                     hst.text()))
@pytest.mark.usefixtures("experiment")
def test_setting_write_period_from_config(wp):
    with reset_config_on_exit():
        qc.config.dataset.write_period = wp

        if isinstance(wp, str):
            with pytest.raises(ValueError):
                Measurement()
        elif wp < 1e-3:
            with pytest.raises(ValueError):
                Measurement()
        else:
            meas = Measurement()
            assert meas.write_period == float(wp)
            meas.register_custom_parameter(name='dummy')
            with meas.run() as datasaver:
                assert datasaver.write_period == float(wp)


@pytest.mark.parametrize("write_in_background", [True, False])
@pytest.mark.usefixtures("experiment")
def test_setting_write_in_background_from_config(write_in_background):
    with reset_config_on_exit():
        qc.config.dataset.write_in_background = write_in_background

        meas = Measurement()
        meas.register_custom_parameter(name='dummy')
        with meas.run() as datasaver:
            ds = datasaver.dataset
            assert isinstance(ds, DataSet)
            assert ds._writer_status.write_in_background is write_in_background


@pytest.mark.usefixtures("experiment")
def test_method_chaining(DAC):
    meas = (
        Measurement()
            .register_parameter(DAC.ch1)
            .register_custom_parameter(name='freqax',
                                       label='Frequency axis',
                                       unit='Hz')
            .add_before_run((lambda: None), ())
            .add_after_run((lambda: None), ())
            .add_subscriber((lambda values, idx, state: None), state=[])
    )


@pytest.mark.usefixtures("experiment")
@settings(deadline=None, suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(words=hst.lists(elements=hst.text(), min_size=4, max_size=10))
def test_enter_and_exit_actions(DAC, words):
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
    Test that subscribers are called at the moment the data is flushed to
    database

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
        ds = datasaver.dataset
        assert isinstance(ds, DataSet)
        assert len(ds.subscribers) == 2

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
            # just enough time has passed between the moment the data is
            # flushed to database and the "state" object (that is passed to
            # subscriber constructor) has been updated by the corresponding
            # subscriber's callback function. At the moment, there is no robust
            # way to ensure this. The reason is that the subscribers have
            # internal queue which is populated via a trigger call from the SQL
            # database, hence from this "main" thread it is difficult to say
            # whether the queue is empty because the subscriber callbacks have
            # already been executed or because the triggers of the SQL database
            # has not been executed yet.
            #
            # In order to overcome this problem, a special decorator is used to
            # wrap the assertions. This is going to ensure that some time is
            # given to the Subscriber threads to finish exhausting the queue.
            @retry_until_does_not_throw(
                exception_class_to_expect=AssertionError, delay=0.5, tries=20)
            def assert_states_updated_from_callbacks():
                assert values_larger_than_7 == values_larger_than_7__expected
                assert list(all_results_dict.keys()) == \
                       [result_index for result_index in range(1, num + 1 + 1)]

            assert_states_updated_from_callbacks()

    # Ensure that after exiting the "run()" context,
    # all subscribers get unsubscribed from the dataset
    assert len(ds.subscribers) == 0

    # Ensure that the triggers for each subscriber
    # have been removed from the database
    get_triggers_sql = "SELECT * FROM sqlite_master WHERE TYPE = 'trigger';"
    triggers = atomic_transaction(ds.conn, get_triggers_sql).fetchall()
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
        ds = datasaver.dataset
        assert isinstance(ds, DataSet)
        subscriber = list(ds.subscribers.values())[0]
        subscriber.min_queue_length = int(len(given_x_vals) + 1)

        for x in given_x_vals:
            datasaver.add_result((DAC.ch1, x))
            # Verify that the subscriber callback is not called yet
            assert collected_x_vals == []

    # Verify that the subscriber callback is finally called
    assert collected_x_vals == given_x_vals


@pytest.mark.serial
@pytest.mark.flaky(reruns=5)
@settings(deadline=None, max_examples=25,
          suppress_health_check=(HealthCheck.function_scoped_fixture,))
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
@settings(max_examples=5, deadline=None,
          suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(breakpoint=hst.integers(min_value=1, max_value=19),
       write_period=hst.floats(min_value=0.1, max_value=1.5),
       set_values=hst.lists(elements=hst.floats(), min_size=20, max_size=20),
       get_values=hst.lists(elements=hst.floats(), min_size=20, max_size=20))
@pytest.mark.usefixtures('set_default_station_to_none')
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

    # More assertions of setpoints, labels and units in the DB!


@pytest.mark.usefixtures('set_default_station_to_none')
def test_datasaver_inst_metadata(experiment, DAC_with_metadata, DMM):
    """
    Check that additional instrument metadata is captured into the dataset snapshot
    """

    station = qc.Station(DAC_with_metadata, DMM)

    meas = Measurement(station=station)
    meas.register_parameter(DAC_with_metadata.ch1)
    meas.register_parameter(DMM.v1, setpoints=(DAC_with_metadata.ch1,))

    with meas.run() as datasaver:
        for set_v in range(10):
            DAC_with_metadata.ch1.set(set_v)
            datasaver.add_result((DAC_with_metadata.ch1, set_v), (DMM.v1, DMM.v1.get()))
    station_snapshot = datasaver.dataset.snapshot['station']
    assert station_snapshot['instruments']['dummy_dac']['metadata'] == {"dac": "metadata"}


def test_exception_happened_during_measurement_is_stored_in_dataset_metadata(
        experiment):
    meas = Measurement()
    meas.register_custom_parameter(name='nodata')

    class SomeMeasurementException(Exception):
        pass

    # `pytest.raises`` is used here instead of custom try-except for convenience
    with pytest.raises(SomeMeasurementException, match='foo') as e:

        with meas.run() as datasaver:
            dataset = datasaver.dataset

            raise SomeMeasurementException('foo')

    metadata = dataset.metadata
    assert "measurement_exception" in metadata

    expected_exception_string = "".join(
        traceback.format_exception(e.type, e.value, e.tb))
    exception_string = metadata["measurement_exception"]
    assert exception_string == expected_exception_string


@pytest.mark.parametrize("bg_writing", [True, False])
@settings(max_examples=10, deadline=None)
@given(N=hst.integers(min_value=2, max_value=500))
@pytest.mark.usefixtures("empty_temp_db")
def test_datasaver_arrays_lists_tuples(bg_writing, N):
    new_experiment('firstexp', sample_name='no sample')

    meas = Measurement()

    meas.register_custom_parameter(name='freqax',
                                   label='Frequency axis',
                                   unit='Hz')
    meas.register_custom_parameter(name='signal',
                                   label='qubit signal',
                                   unit='Majorana number',
                                   setpoints=('freqax',))

    with meas.run(write_in_background=bg_writing) as datasaver:
        freqax = np.linspace(1e6, 2e6, N)
        signal = np.random.randn(N)

        datasaver.add_result(('freqax', freqax), ('signal', signal))

    assert datasaver.points_written == N
    ds = datasaver.dataset
    assert isinstance(ds, DataSet)
    assert not ds.conn.atomic_in_progress

    with meas.run(write_in_background=bg_writing) as datasaver:
        freqax = np.linspace(1e6, 2e6, N)
        signal = np.random.randn(N - 1)

        with pytest.raises(ValueError):
            datasaver.add_result(('freqax', freqax), ('signal', signal))

    meas.register_custom_parameter(name='gate_voltage',
                                   label='Gate tuning potential',
                                   unit='V')
    meas.unregister_parameter('signal')
    meas.register_custom_parameter(name='signal',
                                   label='qubit signal',
                                   unit='Majorana flux',
                                   setpoints=('freqax', 'gate_voltage'))

    # save arrays
    with meas.run(write_in_background=bg_writing) as datasaver:
        freqax = np.linspace(1e6, 2e6, N)
        signal = np.random.randn(N)

        datasaver.add_result(('freqax', freqax),
                             ('signal', signal),
                             ('gate_voltage', 0))

    assert datasaver.points_written == N
    ds = datasaver.dataset
    assert isinstance(ds, DataSet)
    assert not ds.conn.atomic_in_progress

    # save lists
    with meas.run(write_in_background=bg_writing) as datasaver:
        freqax = list(np.linspace(1e6, 2e6, N))
        signal = list(np.random.randn(N))

        datasaver.add_result(('freqax', freqax),
                             ('signal', signal),
                             ('gate_voltage', 0))

    assert datasaver.points_written == N

    # save tuples
    with meas.run(write_in_background=bg_writing) as datasaver:
        freqax = tuple(np.linspace(1e6, 2e6, N))
        signal = tuple(np.random.randn(N))

        datasaver.add_result(('freqax', freqax),
                             ('signal', signal),
                             ('gate_voltage', 0))

    assert datasaver.points_written == N


@pytest.mark.parametrize("bg_writing", [True, False])
@settings(max_examples=10, deadline=None)
@given(N=hst.integers(min_value=2, max_value=500))
@pytest.mark.usefixtures("empty_temp_db")
def test_datasaver_numeric_and_array_paramtype(bg_writing, N):
    """
    Test saving one parameter with 'numeric' paramtype and one parameter with
    'array' paramtype
    """
    new_experiment('firstexp', sample_name='no sample')

    meas = Measurement()

    meas.register_custom_parameter(name='numeric_1',
                                   label='Magnetic field',
                                   unit='T',
                                   paramtype='numeric')
    meas.register_custom_parameter(name='array_1',
                                   label='Alazar signal',
                                   unit='V',
                                   paramtype='array',
                                   setpoints=('numeric_1',))

    signal = np.random.randn(113)

    with meas.run(bg_writing) as datasaver:
        datasaver.add_result(('numeric_1', 3.75), ('array_1', signal))

    assert datasaver.points_written == 1
    ds = datasaver.dataset
    assert isinstance(ds, DataSet)
    data = ds.get_parameter_data(*ds.parameters.split(","))
    assert (data["numeric_1"]["numeric_1"] == np.array([3.75])).all()
    assert np.allclose(data["array_1"]["array_1"], signal)


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.usefixtures("empty_temp_db")
def test_datasaver_numeric_after_array_paramtype(bg_writing):
    """
    Test that passing values for 'array' parameter in `add_result` before
    passing values for 'numeric' parameter works.
    """
    new_experiment('firstexp', sample_name='no sample')

    meas = Measurement()

    meas.register_custom_parameter(name='numeric_1',
                                   label='Magnetic field',
                                   unit='T',
                                   paramtype='numeric')
    meas.register_custom_parameter(name='array_1',
                                   label='Alazar signal',
                                   unit='V',
                                   paramtype='array',
                                   setpoints=('numeric_1',))

    signal = np.random.randn(113)

    with meas.run(write_in_background=bg_writing) as datasaver:
        # it is important that first comes the 'array' data and then 'numeric'
        datasaver.add_result(('array_1', signal), ('numeric_1', 3.75))

    assert datasaver.points_written == 1
    ds = datasaver.dataset
    assert isinstance(ds, DataSet)
    data = ds.get_parameter_data(*ds.parameters.split(","))
    assert (data["numeric_1"]["numeric_1"] == np.array([3.75])).all()
    assert np.allclose(data["array_1"]["array_1"], signal)


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.usefixtures("experiment")
def test_datasaver_foul_input(bg_writing):
    meas = Measurement()

    meas.register_custom_parameter('foul',
                                   label='something unnatural',
                                   unit='Fahrenheit')

    foul_stuff = [qc.Parameter('foul'), {1, 2, 3}]

    with meas.run(bg_writing) as datasaver:
        for ft in foul_stuff:
            with pytest.raises(ValueError):
                datasaver.add_result(('foul', ft))


@settings(max_examples=10, deadline=None)
@given(N=hst.integers(min_value=2, max_value=500))
@pytest.mark.usefixtures("empty_temp_db")
@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.parametrize("storage_type", ['numeric', 'array'])
def test_datasaver_unsized_arrays(N, storage_type, bg_writing):
    new_experiment('firstexp', sample_name='no sample')

    meas = Measurement()

    meas.register_custom_parameter(name='freqax',
                                   label='Frequency axis',
                                   unit='Hz',
                                   paramtype=storage_type)
    meas.register_custom_parameter(name='signal',
                                   label='qubit signal',
                                   unit='Majorana number',
                                   setpoints=('freqax',),
                                   paramtype=storage_type)
    # note that np.array(some_number) is not the same as the number
    # its also not an array with a shape. Check here that we handle it
    # correctly
    with meas.run(write_in_background=bg_writing) as datasaver:
        freqax = np.linspace(1e6, 2e6, N)
        np.random.seed(0)
        signal = np.random.randn(N)
        for i in range(N):
            myfreq = np.array(freqax[i])
            assert myfreq.shape == ()
            mysignal = np.array(signal[i])
            assert mysignal.shape == ()
            datasaver.add_result(('freqax', myfreq), ('signal', mysignal))

    assert datasaver.points_written == N
    ds = datasaver.dataset
    assert isinstance(ds, DataSet)
    loaded_data = ds.get_parameter_data()["signal"]

    np.random.seed(0)
    expected_signal = np.random.randn(N)
    expected_freqax = np.linspace(1e6, 2e6, N)

    if storage_type == 'array':
        expected_freqax = expected_freqax.reshape((N, 1))
        expected_signal = expected_signal.reshape((N, 1))

    assert_allclose(loaded_data['freqax'], expected_freqax)
    assert_allclose(loaded_data['signal'], expected_signal)


@settings(max_examples=5, deadline=None,
          suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(N=hst.integers(min_value=5, max_value=6),
       M=hst.integers(min_value=4, max_value=5),
       seed=hst.integers(min_value=0, max_value=np.iinfo(np.uint32).max))
@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.parametrize("param_type", ['np_array', 'tuple', 'list'])
@pytest.mark.parametrize("storage_type", ['numeric', 'array'])
def test_datasaver_arrayparams(SpectrumAnalyzer, DAC, N, M,
                               param_type, storage_type,
                               seed, bg_writing):
    """
    test that data is stored correctly for array parameters that
    return numpy arrays, lists and tuples. Stored both as arrays and
    numeric
    """

    if param_type == 'list':
        spectrum = SpectrumAnalyzer.listspectrum
        spectrum_name = 'dummy_SA_listspectrum'
    elif param_type == 'tuple':
        spectrum = SpectrumAnalyzer.tuplespectrum
        spectrum_name = 'dummy_SA_tuplespectrum'
    elif param_type == 'np_array':
        spectrum = SpectrumAnalyzer.spectrum
        spectrum_name = 'dummy_SA_spectrum'
    else:
        raise RuntimeError("Invalid storage_type")

    meas = Measurement()

    meas.register_parameter(spectrum, paramtype=storage_type)

    assert len(meas.parameters) == 2
    setpoint_paramspec = meas.parameters['dummy_SA_Frequency']
    spectrum_paramspec = meas.parameters[str(spectrum)]
    assert setpoint_paramspec in meas._interdeps.dependencies[spectrum_paramspec]
    assert meas.parameters[str(spectrum)].type == storage_type
    assert meas.parameters['dummy_SA_Frequency'].type == storage_type

    # Now for a real measurement

    meas = Measurement()

    meas.register_parameter(DAC.ch1)
    meas.register_parameter(spectrum, setpoints=[DAC.ch1], paramtype=storage_type)

    assert len(meas.parameters) == 3

    spectrum.npts = M

    np.random.seed(seed)
    with meas.run(write_in_background=bg_writing) as datasaver:
        for set_v in np.linspace(0, 0.01, N):
            datasaver.add_result((DAC.ch1, set_v),
                                 (spectrum, spectrum.get()))

    if storage_type == 'numeric':
        assert datasaver.points_written == N * M
    elif storage_type == 'array':
        assert datasaver.points_written == N

    np.random.seed(seed)
    expected_dac_data = np.repeat(np.linspace(0, 0.01, N), M)
    expected_freq_axis = np.tile(spectrum.setpoints[0], N)
    expected_output = np.array([spectrum.get() for _ in range(N)]).reshape(
        N * M)

    if storage_type == 'array':
        expected_dac_data = expected_dac_data.reshape(N, M)
        expected_freq_axis = expected_freq_axis.reshape(N, M)
        expected_output = expected_output.reshape(N, M)

    ds = datasaver.dataset
    assert isinstance(ds, DataSet)
    data = ds.get_parameter_data()[spectrum_name]

    assert_allclose(data['dummy_dac_ch1'], expected_dac_data)
    assert_allclose(data['dummy_SA_Frequency'], expected_freq_axis)
    assert_allclose(data[spectrum_name], expected_output)


@settings(max_examples=5, deadline=None,
          suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(N=hst.integers(min_value=5, max_value=500))
@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.parametrize("storage_type", ['numeric', 'array'])
@pytest.mark.usefixtures("experiment")
def test_datasaver_array_parameters_channel(channel_array_instrument,
                                            DAC, N, storage_type,
                                            bg_writing):
    meas = Measurement()

    array_param = channel_array_instrument.A.dummy_array_parameter

    meas.register_parameter(array_param, paramtype=storage_type)

    assert len(meas.parameters) == 2
    dependency_name = 'dummy_channel_inst_ChanA_array_setpoint_param_this_setpoint'
    dep_paramspec = meas.parameters[dependency_name]
    array_paramspec = meas.parameters[str(array_param)]
    assert dep_paramspec in meas._interdeps.dependencies[array_paramspec]
    assert meas.parameters[str(array_param)].type == storage_type
    assert meas.parameters[dependency_name].type == storage_type

    # Now for a real measurement

    meas = Measurement()

    meas.register_parameter(DAC.ch1)
    meas.register_parameter(array_param, setpoints=[DAC.ch1], paramtype=storage_type)

    assert len(meas.parameters) == 3

    M = array_param.shape[0]

    with meas.run(write_in_background=bg_writing) as datasaver:
        for set_v in np.linspace(0, 0.01, N):
            datasaver.add_result((DAC.ch1, set_v),
                                 (array_param, array_param.get()))
    if storage_type == 'numeric':
        n_points_written_expected = N * M
    elif storage_type == 'array':
        n_points_written_expected = N

    assert datasaver.points_written == n_points_written_expected

    expected_params = ('dummy_dac_ch1',
                       dependency_name,
                       'dummy_channel_inst_ChanA_dummy_array_parameter')
    ds = load_by_id(datasaver.run_id)
    assert isinstance(ds, DataSet)
    loaded_data = ds.get_parameter_data()['dummy_channel_inst_ChanA_dummy_array_parameter']
    for param in expected_params:
        if storage_type == 'array':
            expected_shape = (N, M)
        else:
            expected_shape = (N*M, )
        assert loaded_data[param].shape == expected_shape


@settings(max_examples=5, deadline=None,
          suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(n=hst.integers(min_value=5, max_value=500))
@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.parametrize("storage_type", ['numeric', 'array'])
@pytest.mark.usefixtures("experiment")
def test_datasaver_parameter_with_setpoints(channel_array_instrument,
                                            DAC, n, storage_type,
                                            bg_writing):
    random_seed = 1
    chan = channel_array_instrument.A
    param = chan.dummy_parameter_with_setpoints
    chan.dummy_n_points(n)
    chan.dummy_start(0)
    chan.dummy_stop(100)
    meas = Measurement()
    meas.register_parameter(param, paramtype=storage_type)

    assert len(meas.parameters) == 2
    dependency_name = 'dummy_channel_inst_ChanA_dummy_sp_axis'

    dep_ps = meas.parameters[dependency_name]
    param_ps = meas.parameters[str(param)]

    assert dep_ps in meas._interdeps.dependencies[param_ps]
    assert meas.parameters[str(param)].type == storage_type
    assert meas.parameters[dependency_name].type == storage_type

    # Now for a real measurement
    with meas.run(write_in_background=bg_writing) as datasaver:
        # we seed the random number generator
        # so we can test that we get the expected numbers
        np.random.seed(random_seed)
        datasaver.add_result((param, param.get()))
    if storage_type == 'numeric':
        expected_points_written = n
    elif storage_type == 'array':
        expected_points_written = 1

    assert datasaver.points_written == expected_points_written

    expected_params = (dependency_name,
                       'dummy_channel_inst_ChanA_dummy_parameter_with_setpoints')
    ds = load_by_id(datasaver.run_id)
    assert isinstance(ds, DataSet)
    loaded_data = ds.get_parameter_data()
    for param in expected_params:
        data = loaded_data['dummy_channel_inst_ChanA_dummy_parameter_with_setpoints'][param]
        if storage_type == 'array':
            assert data.shape == (expected_points_written, n)
        else:
            assert data.shape == (expected_points_written,)
    assert len(loaded_data) == 1

    subdata = loaded_data[
        'dummy_channel_inst_ChanA_dummy_parameter_with_setpoints']

    expected_dep_data = np.linspace(chan.dummy_start(),
                                    chan.dummy_stop(),
                                    chan.dummy_n_points())
    np.random.seed(random_seed)
    expected_data = np.random.rand(n)
    if storage_type == 'array':
        expected_dep_data = expected_dep_data.reshape((1,
                                                       chan.dummy_n_points()))
        expected_data = expected_data.reshape((1, chan.dummy_n_points()))

    assert_allclose(subdata[dependency_name], expected_dep_data)
    assert_allclose(subdata['dummy_channel_inst_ChanA_'
                            'dummy_parameter_with_setpoints'],
                    expected_data)


@settings(max_examples=5, deadline=None,
          suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(n=hst.integers(min_value=5, max_value=500))
@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.parametrize("storage_type", ['numeric', 'array'])
@pytest.mark.usefixtures("experiment")
def test_datasaver_parameter_with_setpoints_explicitly_expanded(channel_array_instrument,
                                                     DAC, n, storage_type,
                                                     bg_writing):
    random_seed = 1
    chan = channel_array_instrument.A
    param = chan.dummy_parameter_with_setpoints
    chan.dummy_n_points(n)
    chan.dummy_start(0)
    chan.dummy_stop(100)
    meas = Measurement()
    meas.register_parameter(param, paramtype=storage_type)

    assert len(meas.parameters) == 2
    dependency_name = 'dummy_channel_inst_ChanA_dummy_sp_axis'

    dep_ps = meas.parameters[dependency_name]
    param_ps = meas.parameters[str(param)]

    assert dep_ps in meas._interdeps.dependencies[param_ps]
    assert meas.parameters[str(param)].type == storage_type
    assert meas.parameters[dependency_name].type == storage_type

    # Now for a real measurement
    with meas.run(write_in_background=bg_writing) as datasaver:
        # we seed the random number generator
        # so we can test that we get the expected numbers
        np.random.seed(random_seed)
        datasaver.add_result(*expand_setpoints_helper(param))
    if storage_type == 'numeric':
        expected_points_written = n
    elif storage_type == 'array':
        expected_points_written = 1

    assert datasaver.points_written == expected_points_written

    expected_params = (dependency_name,
                       'dummy_channel_inst_ChanA_dummy_parameter_with_setpoints')
    ds = load_by_id(datasaver.run_id)
    assert isinstance(ds, DataSet)
    loaded_data = ds.get_parameter_data()
    for param in expected_params:
        data = loaded_data['dummy_channel_inst_ChanA_dummy_parameter_with_setpoints'][param]
        if storage_type == 'array':
            assert data.shape == (expected_points_written, n)
        else:
            assert data.shape == (expected_points_written,)
    assert len(loaded_data) == 1

    subdata = loaded_data[
        'dummy_channel_inst_ChanA_dummy_parameter_with_setpoints']

    expected_dep_data = np.linspace(chan.dummy_start(),
                                    chan.dummy_stop(),
                                    chan.dummy_n_points())
    np.random.seed(random_seed)
    expected_data = np.random.rand(n)
    if storage_type == 'array':
        expected_dep_data = expected_dep_data.reshape((1,
                                                       chan.dummy_n_points()))
        expected_data = expected_data.reshape((1, chan.dummy_n_points()))

    assert_allclose(subdata[dependency_name], expected_dep_data)
    assert_allclose(subdata['dummy_channel_inst_ChanA_'
                            'dummy_parameter_with_setpoints'],
                    expected_data)


@pytest.mark.usefixtures("experiment")
def test_datasaver_parameter_with_setpoints_partially_expanded_raises(channel_array_instrument, DAC):
    random_seed = 1
    chan = channel_array_instrument.A
    param = chan.dummy_parameter_with_setpoints_2d
    chan.dummy_n_points(10)
    chan.dummy_n_points_2(20)
    chan.dummy_start(0)
    chan.dummy_stop(100)
    chan.dummy_start_2(5)
    chan.dummy_stop_2(7)
    meas = Measurement()
    meas.register_parameter(param)

    sp_param_1 = chan.dummy_sp_axis

    assert len(meas.parameters) == 3
    dependency_name = 'dummy_channel_inst_ChanA_dummy_sp_axis'

    dep_ps = meas.parameters[dependency_name]
    param_ps = meas.parameters[str(param)]

    assert dep_ps in meas._interdeps.dependencies[param_ps]

    with meas.run() as datasaver:
        # we seed the random number generator
        # so we can test that we get the expected numbers
        np.random.seed(random_seed)
        with pytest.raises(ValueError, match="Some of the setpoints of"):
            datasaver.add_result((param, param.get()),
                                 (sp_param_1, sp_param_1.get()))


@pytest.mark.parametrize("bg_writing", [True, False])
@settings(max_examples=5, deadline=None,
          suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(n=hst.integers(min_value=5, max_value=500))
@pytest.mark.usefixtures("experiment")
def test_datasaver_parameter_with_setpoints_complex(channel_array_instrument,
                                                    DAC, n,
                                                    bg_writing):
    random_seed = 1
    chan = channel_array_instrument.A
    param = chan.dummy_parameter_with_setpoints_complex
    chan.dummy_n_points(n)
    chan.dummy_start(0)
    chan.dummy_stop(100)
    meas = Measurement()
    meas.register_parameter(param, paramtype='array')

    assert len(meas.parameters) == 2

    dependency_name = 'dummy_channel_inst_ChanA_dummy_sp_axis'

    dependent_parameter = meas.parameters[str(param)]
    indepdendent_parameter = meas.parameters[dependency_name]

    assert meas._interdeps.dependencies[dependent_parameter] \
           == (indepdendent_parameter, )

    assert dependent_parameter.type == 'array'
    assert indepdendent_parameter.type == 'array'

    # Now for a real measurement
    with meas.run(write_in_background=bg_writing) as datasaver:
        # we seed the random number generator
        # so we can test that we get the expected numbers
        np.random.seed(random_seed)
        datasaver.add_result((param, param.get()))
    assert datasaver.points_written == 1

    ds = load_by_id(datasaver.run_id)
    assert isinstance(ds, DataSet)
    datadict = ds.get_parameter_data()
    assert len(datadict) == 1
    subdata = datadict[
        'dummy_channel_inst_ChanA_dummy_parameter_with_setpoints_complex']
    assert_allclose(subdata[dependency_name],
                    np.linspace(chan.dummy_start(),
                                chan.dummy_stop(),
                                chan.dummy_n_points()).reshape(1, chan.dummy_n_points()))
    np.random.seed(random_seed)
    assert_allclose(subdata['dummy_channel_inst_ChanA_'
                            'dummy_parameter_with_setpoints_complex'],
                    (np.random.rand(n) + 1j * np.random.rand(n)).reshape(1, chan.dummy_n_points()))


@pytest.mark.parametrize("bg_writing", [True, False])
@settings(max_examples=5, deadline=None,
          suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(n=hst.integers(min_value=5, max_value=500))
@pytest.mark.usefixtures("experiment")
def test_datasaver_parameter_with_setpoints_complex_explicitly_expanded(channel_array_instrument,
                                                             DAC, n,
                                                             bg_writing):
    random_seed = 1
    chan = channel_array_instrument.A
    param = chan.dummy_parameter_with_setpoints_complex
    chan.dummy_n_points(n)
    chan.dummy_start(0)
    chan.dummy_stop(100)
    meas = Measurement()
    meas.register_parameter(param, paramtype='array')

    assert len(meas.parameters) == 2

    dependency_name = 'dummy_channel_inst_ChanA_dummy_sp_axis'

    dependent_parameter = meas.parameters[str(param)]
    indepdendent_parameter = meas.parameters[dependency_name]

    assert meas._interdeps.dependencies[dependent_parameter] \
           == (indepdendent_parameter, )

    assert dependent_parameter.type == 'array'
    assert indepdendent_parameter.type == 'array'

    # Now for a real measurement
    with meas.run(write_in_background=bg_writing) as datasaver:
        # we seed the random number generator
        # so we can test that we get the expected numbers
        np.random.seed(random_seed)
        datasaver.add_result(*expand_setpoints_helper(param))
    assert datasaver.points_written == 1

    ds = load_by_id(datasaver.run_id)
    assert isinstance(ds, DataSet)
    datadict = ds.get_parameter_data()
    assert len(datadict) == 1
    subdata = datadict[
        'dummy_channel_inst_ChanA_dummy_parameter_with_setpoints_complex']
    assert_allclose(subdata[dependency_name],
                    np.linspace(chan.dummy_start(),
                                chan.dummy_stop(),
                                chan.dummy_n_points()).reshape(1, chan.dummy_n_points()))
    np.random.seed(random_seed)
    assert_allclose(subdata['dummy_channel_inst_ChanA_'
                            'dummy_parameter_with_setpoints_complex'],
                    (np.random.rand(n) + 1j * np.random.rand(n)).reshape(1, chan.dummy_n_points()))


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.parametrize("storage_type", ['numeric', 'array'])
@pytest.mark.usefixtures("experiment")
def test_datasaver_parameter_with_setpoints_missing_reg_raises(
        channel_array_instrument,
        DAC, storage_type,
        bg_writing):
    """
    Test that if for whatever reason new setpoints are added after
    registering but before adding this raises correctly
    """
    chan = channel_array_instrument.A
    param = chan.dummy_parameter_with_setpoints
    chan.dummy_n_points(11)
    chan.dummy_start(0)
    chan.dummy_stop(10)

    old_setpoints = param.setpoints
    param.setpoints = ()
    meas = Measurement()
    meas.register_parameter(param, paramtype=storage_type)

    param.setpoints = old_setpoints
    with meas.run(write_in_background=bg_writing) as datasaver:
        sp_param_name = 'dummy_channel_inst_ChanA_dummy_sp_axis'
        match = re.escape('Can not add result for parameter '
                          f'{sp_param_name}, no such parameter registered '
                          'with this measurement.')
        with pytest.raises(ValueError, match=match):
            datasaver.add_result(*expand_setpoints_helper(param))

    with meas.run(write_in_background=bg_writing) as datasaver:
        sp_param_name = 'dummy_channel_inst_ChanA_dummy_sp_axis'
        match = re.escape('Can not add result for parameter '
                          f'{sp_param_name}, no such parameter registered '
                          'with this measurement.')
        with pytest.raises(ValueError, match=match):
            datasaver.add_result((param, param.get()))


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.parametrize("storage_type", ['numeric', 'array'])
@pytest.mark.usefixtures("experiment")
def test_datasaver_parameter_with_setpoints_reg_but_missing_validator(
        channel_array_instrument,
        DAC, storage_type,
        bg_writing):
    """
    Test that if for whatever reason the setpoints are removed between
    registering and adding this raises correctly. This tests tests that
    the parameter validator correctly asserts this.
    """
    chan = channel_array_instrument.A
    param = chan.dummy_parameter_with_setpoints
    chan.dummy_n_points(11)
    chan.dummy_start(0)
    chan.dummy_stop(10)

    meas = Measurement()
    meas.register_parameter(param, paramtype=storage_type)

    param.setpoints = ()

    with meas.run(write_in_background=bg_writing) as datasaver:
        with pytest.raises(ValueError, match=r"Shape of output is not"
                                             r" consistent with setpoints."
                                             r" Output is shape "
                                             r"\(<qcodes.instrument.parameter."
                                             r"Parameter: dummy_n_points at "
                                             r"[0-9]+>,\) and setpoints are "
                                             r"shape \(\)', 'getting dummy_"
                                             r"channel_inst_ChanA_dummy_"
                                             r"parameter_with_setpoints"):
            datasaver.add_result(*expand_setpoints_helper(param))

    with meas.run(write_in_background=bg_writing) as datasaver:
        with pytest.raises(ValueError, match=r"Shape of output is not"
                                             r" consistent with setpoints."
                                             r" Output is shape "
                                             r"\(<qcodes.instrument.parameter."
                                             r"Parameter: dummy_n_points at "
                                             r"[0-9]+>,\) and setpoints are "
                                             r"shape \(\)', 'getting dummy_"
                                             r"channel_inst_ChanA_dummy_"
                                             r"parameter_with_setpoints"):
            datasaver.add_result((param, param.get()))


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.parametrize("storage_type", ['numeric', 'array'])
@pytest.mark.usefixtures("experiment")
def test_datasaver_parameter_with_setpoints_reg_but_missing(
        channel_array_instrument,
        DAC, storage_type,
        bg_writing):
    """
    Test that if for whatever reason the setpoints of a QCoDeS parameter are
    removed between registering that parameter with the Measurement and adding
    results for those setpoints, then then datasaver correctly raises.
    """
    chan = channel_array_instrument.A
    param = chan.dummy_parameter_with_setpoints
    chan.dummy_n_points(11)
    chan.dummy_start(0)
    chan.dummy_stop(10)

    someparam = Parameter('someparam', vals=Arrays(shape=(10,)))
    old_setpoints = param.setpoints
    param.setpoints = (old_setpoints[0], someparam)

    meas = Measurement()
    meas.register_parameter(param, paramtype=storage_type)

    param.setpoints = old_setpoints
    with meas.run(write_in_background=bg_writing) as datasaver:
        match = re.escape('Can not add result, some required parameters '
                          'are missing.')
        with pytest.raises(ValueError, match=match):
            datasaver.add_result(*expand_setpoints_helper(param))

    with meas.run(write_in_background=bg_writing) as datasaver:
        match = re.escape('Can not add result, some required parameters '
                          'are missing.')
        with pytest.raises(ValueError, match=match):
            datasaver.add_result((param, param.get()))


@settings(max_examples=5, deadline=None,
          suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(N=hst.integers(min_value=5, max_value=500))
@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize("storage_type", ['numeric', 'array'])
@pytest.mark.parametrize("bg_writing", [True, False])
def test_datasaver_array_parameters_array(channel_array_instrument, DAC, N,
                                          storage_type,
                                          bg_writing):
    """
    Test that storing array parameters inside a loop works as expected
    """
    meas = Measurement()

    array_param = channel_array_instrument.A.dummy_array_parameter

    meas.register_parameter(array_param, paramtype=storage_type)

    assert len(meas.parameters) == 2
    dependency_name = 'dummy_channel_inst_ChanA_array_setpoint_param_this_setpoint'
    dependency_ps = meas.parameters[dependency_name]
    array_param_ps = meas.parameters[str(array_param)]
    assert dependency_ps in meas._interdeps.dependencies[array_param_ps]
    assert meas.parameters[str(array_param)].type == storage_type
    assert meas.parameters[dependency_name].type == storage_type

    # Now for a real measurement

    meas = Measurement()

    meas.register_parameter(DAC.ch1, paramtype='numeric')
    meas.register_parameter(array_param, setpoints=[DAC.ch1], paramtype=storage_type)

    assert len(meas.parameters) == 3

    M = array_param.shape[0]
    dac_datapoints = np.linspace(0, 0.01, N)
    with meas.run(write_in_background=bg_writing) as datasaver:
        for set_v in dac_datapoints:
            datasaver.add_result((DAC.ch1, set_v),
                                 (array_param, array_param.get()))

    if storage_type == 'numeric':
        expected_npoints = N*M
    elif storage_type == 'array':
        expected_npoints = N

    assert datasaver.points_written == expected_npoints
    ds = load_by_id(datasaver.run_id)
    assert isinstance(ds, DataSet)
    loaded_data = ds.get_parameter_data()['dummy_channel_inst_ChanA_dummy_array_parameter']

    data_num = loaded_data['dummy_dac_ch1']
    assert len(data_num) == expected_npoints

    setpoint_arrays = loaded_data[dependency_name]
    data_arrays = loaded_data['dummy_channel_inst_ChanA_dummy_array_parameter']
    assert len(setpoint_arrays) == expected_npoints
    assert len(data_arrays) == expected_npoints

    expected_dac_data = np.repeat(np.linspace(0, 0.01, N), M)
    expected_sp_data = np.tile(array_param.setpoints[0], N)
    expected_output = np.array([array_param.get() for _ in range(N)]).reshape(
        N * M)

    if storage_type == 'array':
        expected_dac_data = expected_dac_data.reshape(N, M)
        expected_sp_data = expected_sp_data.reshape(N, M)
        expected_output = expected_output.reshape(N, M)

    assert_allclose(loaded_data['dummy_dac_ch1'], expected_dac_data)
    assert_allclose(loaded_data[dependency_name],
                    expected_sp_data)
    assert_allclose(loaded_data['dummy_channel_inst_ChanA_dummy_array_parameter'],
                    expected_output)

    if storage_type == 'array':
        for data_array, setpoint_array in zip(data_arrays, setpoint_arrays):
            assert_array_equal(setpoint_array, np.linspace(5, 9, 5))
            assert_array_equal(data_array, np.array([2., 2., 2., 2., 2.]))


@pytest.mark.parametrize("bg_writing", [True, False])
@settings(max_examples=5, deadline=None,
          suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(N=hst.integers(min_value=5, max_value=500))
@pytest.mark.usefixtures("experiment")
def test_datasaver_complex_array_parameters_array(channel_array_instrument,
                                                  DAC, N,
                                                  bg_writing):
    """
    Test that storing a complex array parameters inside a loop with the sqlite
    Array type works as expected
    """
    meas = Measurement()

    array_param = channel_array_instrument.A.dummy_complex_array_parameter

    meas.register_parameter(array_param, paramtype='array')

    assert len(meas.parameters) == 2
    dependency_name = 'dummy_channel_inst_ChanA_this_setpoint'

    dependent_parameter = meas.parameters[str(array_param)]
    indepdendent_parameter = meas.parameters[dependency_name]

    assert meas._interdeps.dependencies[dependent_parameter] \
           == (indepdendent_parameter, )

    assert dependent_parameter.type == 'array'
    assert indepdendent_parameter.type == 'array'

    # Now for a real measurement

    meas = Measurement()

    meas.register_parameter(DAC.ch1, paramtype='numeric')
    meas.register_parameter(array_param, setpoints=[DAC.ch1], paramtype='array')

    assert len(meas.parameters) == 3

    M = array_param.shape[0]
    dac_datapoints = np.linspace(0, 0.01, N)
    with meas.run(write_in_background=bg_writing) as datasaver:
        for set_v in dac_datapoints:
            datasaver.add_result((DAC.ch1, set_v),
                                 (array_param, array_param.get()))
    assert datasaver.points_written == N
    ds = load_by_id(datasaver.run_id)
    assert isinstance(ds, DataSet)
    loaded_data = ds.get_parameter_data()["dummy_channel_inst_ChanA_dummy_complex_array_parameter"]
    data_num = loaded_data['dummy_dac_ch1']
    assert data_num.shape == (N, M)

    param_name = 'dummy_channel_inst_ChanA_dummy_complex_array_parameter'

    setpoint_arrays = loaded_data['dummy_channel_inst_ChanA_this_setpoint']
    data_arrays = loaded_data[param_name]
    assert setpoint_arrays.shape == (N, M)
    assert data_arrays.shape == (N, M)

    for data_array, setpoint_array in zip(data_arrays, setpoint_arrays):
        assert_array_equal(setpoint_array, np.linspace(5, 9, 5))
        assert_array_equal(data_array, np.arange(5) - 1j*np.arange(5))


@pytest.mark.parametrize("bg_writing", [True, False])
def test_datasaver_multidim_array(experiment, bg_writing):  # noqa: F811
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
    expected = {'x1': data[0, :, :],
                'x2': data[1, :, :],
                'y1': data[2, :, :],
                'y2': data[3, :, :]}
    with meas.run(write_in_background=bg_writing) as datasaver:
        datasaver.add_result((str(x1), expected['x1']),
                             (str(x2), expected['x2']),
                             (str(y1), expected['y1']),
                             (str(y2), expected['y2']))

    # We expect one "point" i.e. row in the DB to be written per top-level
    # parameter.
    assert datasaver.points_written == 2
    dataset = load_by_id(datasaver.run_id)
    assert isinstance(dataset, DataSet)
    loaded_data = dataset.get_parameter_data()
    for outerid in ('y1', 'y2'):
        for innerid in ('x1', 'x2', outerid):
            mydata = loaded_data[outerid][innerid]
            assert mydata.shape == (1, size1, size2)
            assert_array_equal(mydata[0], expected[innerid])


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.parametrize("export", [True, False])
def test_datasaver_export(experiment, bg_writing, tmp_path_factory, export, mocker):
    """
    Test export data to csv after measurement ends
    """
    meas = Measurement(experiment)
    size1 = 10
    size2 = 15

    x1 = qc.ManualParameter('x1')
    x2 = qc.ManualParameter('x2')
    y1 = qc.ManualParameter('y1')
    y2 = qc.ManualParameter('y2')

    meas.register_parameter(x1, paramtype='array')
    meas.register_parameter(x2, paramtype='array')
    meas.register_parameter(y1, setpoints=[x1, x2], paramtype='array')
    meas.register_parameter(y2, setpoints=[x1, x2], paramtype='array')
    data = np.random.rand(4, size1, size2)
    expected = {'x1': data[0, :, :],
                'x2': data[1, :, :],
                'y1': data[2, :, :],
                'y2': data[3, :, :]}

    tmp_path = tmp_path_factory.mktemp("export_from_config")
    path = str(tmp_path)

    mock_type = mocker.patch("qcodes.dataset.data_set_protocol.get_data_export_type")
    mock_path = mocker.patch("qcodes.dataset.data_set_protocol.get_data_export_path")
    mock_automatic = mocker.patch(
        "qcodes.dataset.measurements.get_data_export_automatic"
    )

    mock_type.return_value = DataExportType.CSV
    mock_path.return_value = path
    mock_automatic.return_value = export
    with meas.run(write_in_background=bg_writing) as datasaver:
        datasaver.add_result(
            (str(x1), expected["x1"]),
            (str(x2), expected["x2"]),
            (str(y1), expected["y1"]),
            (str(y2), expected["y2"]),
        )
    if export:
        assert os.listdir(path) == [
            f"qcodes_{datasaver.dataset.captured_run_id}_{datasaver.dataset.guid}.csv"
        ]
    else:
        assert os.listdir(path) == []


@pytest.mark.parametrize("bg_writing", [True, False])
def test_datasaver_multidim_numeric(experiment, bg_writing):
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
    with meas.run(write_in_background=bg_writing) as datasaver:
        datasaver.add_result((str(x1), data[0, :, :]),
                             (str(x2), data[1, :, :]),
                             (str(y1), data[2, :, :]),
                             (str(y2), data[3, :, :]))
    # The factor of 2 is due to there being 2 top-level params
    assert datasaver.points_written == 2 * (size1 * size2)
    dataset = load_by_id(datasaver.run_id)
    assert isinstance(dataset, DataSet)
    all_data = dataset.get_parameter_data()
    for outer in ('y1', 'y2'):
        for inner in ('x1', 'x2', outer):
            mydata = all_data[outer][inner]
            assert mydata.shape == (size1 * size2, )
            assert mydata.dtype == np.float64


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.usefixtures("experiment")
def test_datasaver_multidimarrayparameter_as_array(SpectrumAnalyzer,
                                                   bg_writing):
    """
    Test that inserting multidim Arrrayparameters as array works as expected
    """
    array_param = SpectrumAnalyzer.multidimspectrum
    meas = Measurement()
    meas.register_parameter(array_param, paramtype='array')
    assert len(meas.parameters) == 4
    inserted_data = array_param.get()
    with meas.run(write_in_background=bg_writing) as datasaver:
        datasaver.add_result((array_param, inserted_data))

    assert datasaver.points_written == 1
    ds = load_by_id(datasaver.run_id)
    assert isinstance(ds, DataSet)
    expected_shape = (1, 100, 50, 20)
    loaded_data = ds.get_parameter_data()
    for i in range(3):
        data = loaded_data['dummy_SA_multidimspectrum'][f'dummy_SA_Frequency{i}']
        aux_shape = list(expected_shape)
        aux_shape.pop(i+1)

        assert data.shape == expected_shape
        for j in range(aux_shape[1]):
            for k in range(aux_shape[2]):
                # todo There should be a simpler way of doing this
                if i == 0:
                    mydata = data[0, :, j, k]
                if i == 1:
                    mydata = data[0, j, :, k]
                if i == 2:
                    mydata = data[0, j, k, :]
                assert_array_equal(mydata,
                                   np.linspace(array_param.start,
                                               array_param.stop,
                                               array_param.npts[i]))


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.usefixtures("experiment")
def test_datasaver_multidimarrayparameter_as_numeric(SpectrumAnalyzer,
                                                     bg_writing):
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
    with meas.run(write_in_background=bg_writing) as datasaver:
        datasaver.add_result((array_param, inserted_data))

    assert datasaver.points_written == points_expected
    ds = load_by_id(datasaver.run_id)
    assert isinstance(ds, DataSet)
    # check setpoints

    expected_setpoints_vectors = (np.linspace(array_param.start,
                                              array_param.stop,
                                              array_param.npts[i]) for i in
                                  range(dims))
    expected_setpoints_matrix = np.meshgrid(*expected_setpoints_vectors,
                                            indexing='ij')
    expected_setpoints = tuple(
        setpoint_array.ravel() for setpoint_array in expected_setpoints_matrix)

    loaded_data = ds.get_parameter_data()

    for i in range(dims):
        data = loaded_data['dummy_SA_multidimspectrum'][f'dummy_SA_Frequency{i}']
        assert len(data) == points_expected
        assert_allclose(data.squeeze(),
                        expected_setpoints[i])
    data = loaded_data['dummy_SA_multidimspectrum']['dummy_SA_multidimspectrum'].squeeze()
    assert_allclose(data, inserted_data.ravel())


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.usefixtures("experiment")
def test_datasaver_multi_parameters_scalar(channel_array_instrument,
                                           bg_writing):
    """
    Test that we can register multiparameters that are scalar.
    """
    meas = Measurement()
    param = channel_array_instrument.A.dummy_scalar_multi_parameter
    meas.register_parameter(param)
    assert len(meas.parameters) == len(param.shapes)
    assert set(meas.parameters.keys()) == set(param.full_names)

    with meas.run(write_in_background=bg_writing) as datasaver:
        datasaver.add_result((param, param()))

    assert datasaver.points_written == 2
    ds = load_by_id(datasaver.run_id)
    assert isinstance(ds, DataSet)
    assert ds.get_parameter_data()['dummy_channel_inst_ChanA_thisparam']['dummy_channel_inst_ChanA_thisparam'] == np.array([[0]])
    assert ds.get_parameter_data()['dummy_channel_inst_ChanA_thatparam']['dummy_channel_inst_ChanA_thatparam'] == np.array([[1]])


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.usefixtures("experiment")
def test_datasaver_multi_parameters_array(channel_array_instrument,
                                          bg_writing):
    """
    Test that we can register multiparameters that are array like.
    """
    meas = Measurement()
    param = channel_array_instrument.A.dummy_multi_parameter
    meas.register_parameter(param)
    assert len(meas.parameters) == 3  # two params + 1D identical setpoints
    param_names = ('dummy_channel_inst_ChanA_multi_setpoint_param_this_setpoint',
                   'dummy_channel_inst_ChanA_multi_setpoint_param_this',
                   'dummy_channel_inst_ChanA_multi_setpoint_param_that')
    assert set(meas.parameters.keys()) == set(param_names)
    this_ps = meas.parameters[param_names[1]]
    that_ps = meas.parameters[param_names[2]]
    sp_ps = meas.parameters[param_names[0]]
    assert sp_ps in meas._interdeps.dependencies[this_ps]
    assert sp_ps in meas._interdeps.dependencies[that_ps]

    with meas.run(write_in_background=bg_writing) as datasaver:
        datasaver.add_result((param, param()))
    assert datasaver.points_written == 2 * 5
    ds = load_by_id(datasaver.run_id)
    assert isinstance(ds, DataSet)
    setpts = np.arange(5, 10)

    np.testing.assert_array_equal(ds.get_parameter_data()[param_names[1]][param_names[0]], setpts)
    np.testing.assert_array_equal(ds.get_parameter_data()[param_names[2]][param_names[0]], setpts)

    this_read_data = ds.get_parameter_data()[param_names[1]][param_names[1]]
    that_read_data = ds.get_parameter_data()[param_names[2]][param_names[2]]
    np.testing.assert_array_equal(this_read_data, np.zeros(5))
    np.testing.assert_array_equal(that_read_data, np.ones(5))


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.usefixtures("experiment")
def test_datasaver_2d_multi_parameters_array(channel_array_instrument,
                                             bg_writing):
    """
    Test that we can register multiparameters that are array like and 2D.
    """
    sp_name_1 = "dummy_channel_inst_ChanA_multi_2d_setpoint_param_this_setpoint"
    sp_name_2 = 'dummy_channel_inst_ChanA_multi_2d_setpoint_param_that_setpoint'
    p_name_1 = 'dummy_channel_inst_ChanA_this'
    p_name_2 = 'dummy_channel_inst_ChanA_that'
    from functools import reduce

    meas = Measurement()
    param = channel_array_instrument.A.dummy_2d_multi_parameter
    meas.register_parameter(param)
    assert len(meas.parameters) == 4  # two params + 2D identical setpoints
    param_names = (sp_name_1,
                   sp_name_2,
                   p_name_1,
                   p_name_2)
    assert set(meas.parameters.keys()) == set(param_names)
    this_ps = meas.parameters[p_name_1]
    that_ps = meas.parameters[p_name_2]
    this_sp_ps = meas.parameters[sp_name_1]
    that_sp_ps = meas.parameters[sp_name_2]
    assert that_sp_ps in meas._interdeps.dependencies[this_ps]
    assert that_sp_ps in meas._interdeps.dependencies[that_ps]
    assert this_sp_ps in meas._interdeps.dependencies[this_ps]
    assert this_sp_ps in meas._interdeps.dependencies[that_ps]

    with meas.run(write_in_background=bg_writing) as datasaver:
        datasaver.add_result((param, param()))

    assert datasaver.points_written == 2 * 15
    ds = load_by_id(datasaver.run_id)

    # 30 points in each setpoint value list
    this_sp_val = np.array(reduce(list.__add__, [[n]*3 for n in range(5, 10)], []))
    that_sp_val = np.array(reduce(list.__add__, [[n] for n in range(9, 12)], []) * 5)

    assert isinstance(ds, DataSet)
    np.testing.assert_array_equal(
        ds.get_parameter_data()[p_name_1][sp_name_1],
        this_sp_val
    )
    np.testing.assert_array_equal(
        ds.get_parameter_data()[p_name_1][sp_name_2],
        that_sp_val
    )
    np.testing.assert_array_equal(
        ds.get_parameter_data()[p_name_2][sp_name_1],
        this_sp_val
    )
    np.testing.assert_array_equal(
        ds.get_parameter_data()[p_name_2][sp_name_2],
        that_sp_val
    )

    this_read_data = ds.get_parameter_data()[p_name_1][p_name_1]
    that_read_data = ds.get_parameter_data()[p_name_2][p_name_2]
    assert len(this_read_data) == 15
    assert len(that_read_data) == 15

    np.testing.assert_array_equal(this_read_data, np.zeros(15))
    np.testing.assert_array_equal(that_read_data, np.ones(15))


@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.parametrize("storage_type", ['numeric', 'array'])
@settings(deadline=None)
@given(Ns=hst.lists(hst.integers(2, 10), min_size=2, max_size=5))
def test_datasaver_arrays_of_different_length(storage_type, Ns, bg_writing):
    """
    Test that we can save arrays of different length in a single call to
    datasaver.add_result
    """

    no_of_signals = len(Ns)

    meas = Measurement()
    meas.register_custom_parameter('temperature',
                                   paramtype='numeric',
                                   label='Temperature',
                                   unit='K')
    for n in range(no_of_signals):
        meas.register_custom_parameter(f'freqs{n}', paramtype=storage_type)
        meas.register_custom_parameter(f'signal{n}',
                                       paramtype=storage_type,
                                       setpoints=(f'freqs{n}', 'temperature'))

    with meas.run(write_in_background=bg_writing) as datasaver:
        result_t = ("temperature", 70)
        result_freqs = list(
            (f"freqs{n}", np.linspace(0, 1, Ns[n])) for n in range(no_of_signals)
        )
        result_sigs = list(
            (f"signal{n}", np.random.randn(Ns[n])) for n in range(no_of_signals)
        )
        full_result = tuple(result_freqs + result_sigs + [result_t])
        datasaver.add_result(*full_result)

    ds = load_by_id(datasaver.run_id)

    assert isinstance(ds, DataSet)
    data = ds.get_parameter_data()

    assert list(data.keys()) == [f'signal{n}' for n in range(no_of_signals)]
    for n in range(no_of_signals):
        assert (data[f'signal{n}']['temperature'] == np.array([70]*(Ns[n]))).all()


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.usefixtures("experiment")
def test_save_complex_num(complex_num_instrument, bg_writing):
    """
    Test that we can save various parameters mixed with complex parameters
    """

    # scalar complex parameter
    setparam = complex_num_instrument.setpoint
    param = complex_num_instrument.complex_num
    # array parameter
    arrayparam = complex_num_instrument.some_array
    # complex array parameter
    complexarrayparam = complex_num_instrument.some_complex_array
    some_complex_array_setpoints = complex_num_instrument.some_complex_array_setpoints

    meas = Measurement()
    meas.register_parameter(setparam, paramtype='numeric')
    meas.register_parameter(param, paramtype='complex', setpoints=(setparam,))
    meas.register_parameter(arrayparam, paramtype='array',
                            setpoints=(setparam,))

    meas.register_parameter(some_complex_array_setpoints, paramtype='numeric')
    meas.register_parameter(complexarrayparam, paramtype='complex',
                            setpoints=(setparam, some_complex_array_setpoints))

    with meas.run(write_in_background=bg_writing) as datasaver:
        for i in range(10):
            setparam.set(i)
            datasaver.add_result((setparam, setparam()),
                                 (param, param()),
                                 *expand_setpoints_helper(arrayparam),
                                 (some_complex_array_setpoints, some_complex_array_setpoints.get()),
                                 (complexarrayparam, complexarrayparam.get()))

    ds = datasaver.dataset
    assert isinstance(ds, DataSet)
    data = ds.get_parameter_data()

    # scalar complex parameter
    setpoints_num = data['dummy_channel_inst_complex_num'][
        'dummy_channel_inst_setpoint']
    data_num = data['dummy_channel_inst_complex_num'][
        'dummy_channel_inst_complex_num']

    assert_allclose(setpoints_num, np.arange(10))
    assert_allclose(data_num, np.arange(10) + 1j*np.arange(10))

    # array parameter
    setpoints1_array = data['dummy_channel_inst_some_array'][
        'dummy_channel_inst_setpoint']
    assert_allclose(setpoints1_array, np.repeat(np.arange(10), 5).reshape(10, 5))

    setpoints2_array = data['dummy_channel_inst_some_array'][
        'dummy_channel_inst_some_array_setpoints']

    assert_allclose(setpoints2_array, np.tile(np.arange(5), 10).reshape(10, 5))

    array_data = data['dummy_channel_inst_some_array'][
        'dummy_channel_inst_some_array']

    assert_allclose(array_data, np.ones((10, 5)))

    # complex array parameter
    setpoints1_array = data['dummy_channel_inst_some_complex_array'][
        'dummy_channel_inst_setpoint']
    assert_allclose(setpoints1_array, np.repeat(np.arange(10), 5))

    setpoints2_array = data['dummy_channel_inst_some_complex_array'][
        'dummy_channel_inst_some_complex_array_setpoints']

    assert_allclose(setpoints2_array, np.tile(np.arange(5), 10))

    array_data = data['dummy_channel_inst_some_complex_array'][
        'dummy_channel_inst_some_complex_array']

    assert_allclose(array_data, np.ones(50)+1j*np.ones(50))


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.usefixtures("experiment")
def test_save_and_reload_complex_standalone(complex_num_instrument,
                                            bg_writing):
    param = complex_num_instrument.complex_num
    complex_num_instrument.setpoint(1)
    p = qc.instrument.parameter.Parameter(
        'test',
        set_cmd=None,
        get_cmd=lambda: 1+1j,
        vals=qc.utils.validators.ComplexNumbers())
    meas = qc.dataset.measurements.Measurement()
    meas.register_parameter(param)
    pval = param.get()
    with meas.run(write_in_background=bg_writing) as datasaver:
        datasaver.add_result((param, pval))
    ds = datasaver.dataset
    assert isinstance(ds, DataSet)
    data = ds.get_parameter_data()
    data_num = data['dummy_channel_inst_complex_num'][
        'dummy_channel_inst_complex_num']
    assert_allclose(data_num, 1 + 1j)


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.usefixtures("experiment")
def test_save_complex_num_setpoints(complex_num_instrument, bg_writing):
    """
    Test that we can save a parameter with complex setpoints
    """
    setparam = complex_num_instrument.complex_setpoint
    param = complex_num_instrument.real_part
    meas = Measurement()
    meas.register_parameter(setparam, paramtype='complex')
    meas.register_parameter(param, paramtype='numeric', setpoints=(setparam,))

    with meas.run(write_in_background=bg_writing) as datasaver:
        for i in range(10):
            setparam.set(i+1j*i)
            datasaver.add_result((setparam, setparam()),
                                 (param, param()))
    ds = datasaver.dataset
    assert isinstance(ds, DataSet)
    data = ds.get_parameter_data()
    setpoints_num = data['dummy_channel_inst_real_part'][
        'dummy_channel_inst_complex_setpoint']
    data_num = data['dummy_channel_inst_real_part'][
        'dummy_channel_inst_real_part']

    assert_allclose(setpoints_num, np.arange(10) + 1j*np.arange(10))
    assert_allclose(data_num, np.arange(10))


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.usefixtures("experiment")
def test_save_complex_num_setpoints_array(complex_num_instrument, bg_writing):
    """
    Test that we can save an array parameter with complex setpoints
    """

    setparam = complex_num_instrument.complex_setpoint
    param = complex_num_instrument.some_array

    meas = Measurement()
    meas.register_parameter(setparam, paramtype='complex')
    meas.register_parameter(param, paramtype='array', setpoints=(setparam,))

    with meas.run(write_in_background=bg_writing) as datasaver:
        for i in range(10):
            setparam.set(i+1j*i)
            datasaver.add_result((setparam, setparam()),
                                 *expand_setpoints_helper(param))
    ds = datasaver.dataset
    assert isinstance(ds, DataSet)
    data = ds.get_parameter_data()
    setpoints1 = data['dummy_channel_inst_some_array'][
        'dummy_channel_inst_complex_setpoint']
    setpoints2 = data['dummy_channel_inst_some_array'][
        'dummy_channel_inst_some_array_setpoints']
    data_num = data['dummy_channel_inst_some_array'][
        'dummy_channel_inst_some_array']

    assert_allclose(setpoints1, np.repeat(np.arange(10) +
                                          1j*np.arange(10), 5).reshape((10, 5)))
    assert_allclose(setpoints2, np.tile(np.arange(5), 10).reshape((10, 5)))

    assert_allclose(data_num, np.ones((10, 5)))


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.usefixtures("experiment")
def test_save_complex_as_num_raises(complex_num_instrument, bg_writing):
    setparam = complex_num_instrument.setpoint
    param = complex_num_instrument.complex_num
    meas = Measurement()
    meas.register_parameter(setparam, paramtype='numeric')
    meas.register_parameter(param, paramtype='numeric', setpoints=(setparam,))

    expected_msg = ('Parameter dummy_channel_inst_complex_num is of '
                    'type "numeric", but got a result of '
                    'type complex128')

    with meas.run(write_in_background=bg_writing) as datasaver:
        setparam.set(0)
        with pytest.raises(ValueError, match=expected_msg):
            datasaver.add_result((setparam, setparam()),
                                 (param, param()))


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.usefixtures("experiment")
def test_save_numeric_as_complex_raises(complex_num_instrument, bg_writing):
    setparam = complex_num_instrument.setpoint
    param = complex_num_instrument.complex_num
    meas = Measurement()
    meas.register_parameter(setparam, paramtype='numeric')
    meas.register_parameter(param, paramtype='complex', setpoints=(setparam,))

    expected_msg = ('Parameter dummy_channel_inst_complex_num is of '
                    'type "complex", but got a result of type int')

    with meas.run(write_in_background=bg_writing) as datasaver:
        setparam.set(0)
        with pytest.raises(ValueError, match=expected_msg):
            datasaver.add_result((setparam, setparam()),
                                 (param, setparam()))


def test_parameter_inference(channel_array_instrument):
    chan = channel_array_instrument.channels[0]
    # default values
    assert Measurement._infer_paramtype(chan.temperature, None) is None
    assert Measurement._infer_paramtype(chan.dummy_array_parameter,
                                        None) == 'array'
    assert Measurement._infer_paramtype(chan.dummy_parameter_with_setpoints,
                                        None) == 'array'
    assert Measurement._infer_paramtype(chan.dummy_multi_parameter,
                                        None) is None
    assert Measurement._infer_paramtype(chan.dummy_scalar_multi_parameter,
                                        None) is None
    assert Measurement._infer_paramtype(chan.dummy_2d_multi_parameter,
                                        None) is None
    assert Measurement._infer_paramtype(chan.dummy_text,
                                        None) == 'text'
    assert Measurement._infer_paramtype(chan.dummy_complex,
                                        None) == 'complex'

    # overwrite the default with sensible alternatives
    assert Measurement._infer_paramtype(chan.dummy_array_parameter,
                                        'numeric') == 'numeric'
    assert Measurement._infer_paramtype(chan.dummy_parameter_with_setpoints,
                                        'numeric') == 'numeric'
    assert Measurement._infer_paramtype(chan.dummy_multi_parameter,
                                        'array') == 'array'
    assert Measurement._infer_paramtype(chan.dummy_2d_multi_parameter,
                                        'array') == 'array'


@pytest.mark.usefixtures("experiment")
def test_load_legacy_files_2D():
    location = '../fixtures/2018-01-17/#002_2D_test_15-43-14'
    directory = os.path.dirname(__file__)
    full_location = os.path.join(directory, location)
    run_ids = import_dat_file(full_location)
    run_id = run_ids[0]
    data = load_by_id(run_id)
    assert isinstance(data, DataSet)
    assert data.parameters == 'dac_ch1_set,dac_ch2_set,dmm_voltage'
    assert data.number_of_results == 36
    expected_names = ['dac_ch1_set', 'dac_ch2_set', 'dmm_voltage']
    expected_labels = ['Gate ch1', 'Gate ch2', 'Gate voltage']
    expected_units = ['V', 'V', 'V']
    expected_depends_on = ['', '', 'dac_ch1_set, dac_ch2_set']
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


@pytest.mark.usefixtures("experiment")
def test_load_legacy_files_1D():
    location = '../fixtures/2018-01-17/#001_testsweep_15-42-57'
    dir = os.path.dirname(__file__)
    full_location = os.path.join(dir, location)
    run_ids = import_dat_file(full_location)
    run_id = run_ids[0]
    data = load_by_id(run_id)
    assert isinstance(data, DataSet)
    assert data.parameters == 'dac_ch1_set,dmm_voltage'
    assert data.number_of_results == 201
    expected_names = ['dac_ch1_set', 'dmm_voltage']
    expected_labels = ['Gate ch1', 'Gate voltage']
    expected_units = ['V', 'V']
    expected_depends_on = ['', 'dac_ch1_set']
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


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.usefixtures("experiment")
def test_adding_parents(bg_writing, DAC):
    """
    Test that we can register a DataSet as the parent of another DataSet
    as created by the Measurement
    """

    # The narrative of the test is that we do a measurement once, then learn
    # from the result of that where to measure next. We want to annotate the
    # second run as having the first run as predecessor


    meas = (Measurement()
            .register_parameter(DAC.ch1)
            .register_parameter(DAC.ch2, setpoints=[DAC.ch1]))

    with meas.run(write_in_background=bg_writing) as datasaver:
        datasaver.add_result((DAC.ch1, 0), (DAC.ch2, 1))

    parent_ds = datasaver.dataset

    meas = (Measurement()
            .register_parameter(DAC.ch1)
            .register_parameter(DAC.ch2, setpoints=[DAC.ch1])
            .register_parent(parent=parent_ds, link_type="predecessor"))

    with meas.run(write_in_background=bg_writing) as datasaver:
        datasaver.add_result((DAC.ch1, 1), (DAC.ch2, 2))

    child_ds = datasaver.dataset

    ds_links = child_ds.parent_dataset_links

    assert len(ds_links) == 1
    assert ds_links[0].tail == parent_ds.guid
    assert ds_links[0].head == child_ds.guid
    assert ds_links[0].edge_type == "predecessor"
