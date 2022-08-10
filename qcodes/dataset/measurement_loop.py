import logging
import threading
import traceback
from datetime import datetime
from time import perf_counter, sleep
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np

from qcodes import config as qcodes_config
from qcodes.dataset.descriptions.detect_shapes import detect_shape_of_measurement
from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.descriptions.versioning import serialization as serial
from qcodes.dataset.descriptions.versioning.converters import new_to_old
from qcodes.dataset.do_nd import AbstractSweep
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.sqlite.queries import add_parameter, update_run_description
from qcodes.instrument.base import InstrumentBase
from qcodes.instrument.parameter import _BaseParameter, DelegateParameter, MultiParameter, Parameter
from qcodes.instrument.sweep_values import SweepValues
from qcodes.parameters.parameter_base import ParameterBase
from qcodes.station import Station
from qcodes.utils.dataset.doNd import AbstractSweep, ActionsT
from qcodes.utils.helpers import (
    PerformanceTimer,
    directly_executed_from_cell,
    get_last_input_cells,
    using_ipython,
)

RAW_VALUE_TYPES = (float, int, bool, np.ndarray, np.integer, np.floating, np.bool_, type(None))


class DatasetHandler:
    """Handler for a single DataSet (with Measurement and Runner)"""
    def __init__(self, measurement_loop, name='results'):
        self.measurement_loop = measurement_loop
        self.name = name

        self.initialized = False
        self.datasaver = None
        self.runner = None
        self.measurement = None
        self.dataset = None

        # Key: action_index
        # Values:
        # - parameter
        # - dataset_parameter (differs from 'parameter' when multiple share same name)
        # - latest_value
        self.setpoint_list = dict()

        self.measurement_list = dict()
        # Dict with key being action_index and value is a dict containing
        # - parameter
        # - setpoints_action_indices
        # - setpoint_parameters
        # - shape
        # - unstored_results - list where each element contains (*setpoints, measurement_value)
        # - latest_value

        self.initialize()

    def initialize(self):
        # Once initialized, no new parameters can be added
        assert not self.initialized, "Cannot initialize twice"

        # Create Measurement
        self.measurement = Measurement(name=self.name)

        # Create measurement Runner
        self.runner = self.measurement.run(allow_empty_dataset=True)

        # Create measurement Dataset
        self.datasaver = self.runner.__enter__()
        self.dataset = self.datasaver.dataset

        self.initialized = True

    def finalize(self):
        self.datasaver.flush_data_to_database()

    def _ensure_unique_parameter(self, parameter_info, setpoint, max_idx=99):
        """Ensure parameters have unique names"""
        if setpoint:
            parameter_list = self.setpoint_list
        else:
            parameter_list = self.measurement_list

        parameter_names = [
            param_info['dataset_parameter'].name
            for param_info in parameter_list.values()
            if 'dataset_parameter' in param_info
        ]

        parameter_name = parameter_info['parameter'].name
        if parameter_name not in parameter_names:
            parameter_info['dataset_parameter'] = parameter_info['parameter']
        else:
            for idx in range(1, max_idx):
                parameter_idx_name = f'{parameter_name}_{idx}'
                if parameter_idx_name not in parameter_names:
                    parameter_name = parameter_idx_name
                    break
            else:
                raise OverflowError(
                    f'All parameter names {parameter_name}_{{idx}} up to idx {max_idx} are taken'
                    )
            # Create a delegate parameter with modified name
            delegate_parameter = DelegateParameter(
                name=parameter_name,
                source=parameter_info['parameter']
            )
            parameter_info['dataset_parameter'] = delegate_parameter

    def create_measurement_info(
        self, action_indices, parameter, name=None, label=None, unit=None
    ):
        if parameter is None:
            assert name is not None
            parameter = Parameter(name=name, label=label, unit=unit)
        elif {name, label, unit} != {None, }:
            overwrite_attrs = {
                'name': name,
                'label': label,
                'unit': unit
            }
            overwrite_attrs = {key: val for key, val in overwrite_attrs.items() if val is not None}
            parameter = DelegateParameter(
                source=parameter,
                **overwrite_attrs
            )

        setpoints_action_indices = []
        for k in range(len(action_indices) + 1):
            if action_indices[:k] in self.setpoint_list:
                setpoints_action_indices.append(action_indices[:k])

        measurement_info = {
            'parameter': parameter,
            'setpoints_action_indices': setpoints_action_indices,
            'shape': self.measurement_loop.loop_shape,
            'unstored_results': [],
            'registered': False
        }

        return measurement_info

    def register_new_measurement(
        self,
        action_indices,
        parameter,
        name: str = None,
        label: str = None,
        unit: str = None
    ):
        measurement_info = self.create_measurement_info(
            action_indices=action_indices,
            parameter=parameter,
            name=name,
            label=label,
            unit=unit
        )
        self.measurement_list[action_indices] = measurement_info

        # Add new measurement parameter
        self._update_interdependencies()

    def add_measurement_result(
        self,
        action_indices,
        result,
        parameter=None,
        name: str = None,
        label: str = None,
        unit: str = None,
        ):
        """Store single measurement result

        This method is called from type-specific methods, such as
        ``_measure_value``, ``_measure_parameter``, etc.
        """
        if parameter is None and name is None:
            raise SyntaxError(
                "When adding a measurement result, must provide either a "
                "parameter or name"
            )

        # Get parameter data array, creating a new one if necessary
        if action_indices not in self.measurement_list:
            self.register_new_measurement(
                action_indices=action_indices,
                parameter=parameter,
                name=name,
                label=label,
                unit=unit
            )

        measurement_info = self.measurement_list[action_indices]

        if name is None and parameter is not None:
            name = parameter.name
        if name != measurement_info['parameter'].name:
            raise SyntaxError(
                f'Provided name {name} must match that of previous measurement '
                f"{measurement_info['parameter'].name}"
            )

        # Store result
        setpoints = [
            self.setpoint_list[action_indices]['latest_value']
            for action_indices in measurement_info['setpoints_action_indices']
        ]
        parameters = (
            *measurement_info["setpoint_parameters"],
            measurement_info["dataset_parameter"],
        )
        result_with_setpoints = tuple(zip(parameters, (*setpoints, result)))
        self.datasaver.add_result(*result_with_setpoints)

        # Also store in measurement_info
        measurement_info["latest_value"] = result

    def _update_interdependencies(self):
        dataset = self.datasaver.dataset

        # Get previous paramspecs
        previous_paramspecs = dataset._rundescriber.interdeps.paramspecs
        previous_paramspec_names = [spec.name for spec in previous_paramspecs]

        # Register all new setpoints parameters in Measurement
        for setpoint_info in self.setpoint_list.values():
            if setpoint_info['registered']:
                # Already registered
                continue

            self._ensure_unique_parameter(setpoint_info, setpoint=True)
            self.measurement.register_parameter(setpoint_info['dataset_parameter'])
            setpoint_info['registered'] = True

        # Register all measurement parameters in Measurement
        for measurement_info in self.measurement_list.values():
            if measurement_info['registered']:
                # Already registered
                continue

            # Determine setpoint_parameters for each measurement_parameter
            for measurement_info in self.measurement_list.values():
                measurement_info['setpoint_parameters'] = tuple(
                    self.setpoint_list[action_indices]['dataset_parameter']
                    for action_indices in measurement_info['setpoints_action_indices']
                )

            self._ensure_unique_parameter(measurement_info, setpoint=False)
            self.measurement.register_parameter(
                measurement_info['dataset_parameter'],
                setpoints=measurement_info['setpoint_parameters']
            )
            measurement_info['registered'] = True
            self.measurement.set_shapes(
                detect_shape_of_measurement(
                    (measurement_info["dataset_parameter"],), measurement_info["shape"]
                )
            )

        # Update DataSaver
        self.datasaver._interdeps = self.measurement._interdeps

        # Update DataSet
        # Generate new paramspecs with matching RunDescriber
        dataset._rundescriber = RunDescriber(
            self.measurement._interdeps, shapes=self.measurement._shapes
        )
        paramspecs = new_to_old(dataset._rundescriber.interdeps).paramspecs

        # Add new paramspecs
        for spec in paramspecs:
            if spec.name not in previous_paramspec_names:
                add_parameter(
                    spec,
                    conn=dataset.conn,
                    run_id=dataset.run_id,
                    insert_into_results_table=True,
                )

        desc_str = serial.to_json_for_storage(dataset.description)

        update_run_description(dataset.conn, dataset.run_id, desc_str)

        # Update dataset cache
        cache_data = self.dataset._cache._data
        interdeps_empty_dict = dataset._rundescriber.interdeps._empty_data_dict()
        for key, val in interdeps_empty_dict.items():
            cache_data.setdefault(key, val)

class MeasurementLoop:
    """Class to perform measurements

    Args:
        name: Measurement name, also used as the dataset name
        force_cell_thread: Enforce that the measurement has been started from a
            separate thread if it has been directly executed from an IPython
            cell/prompt. This is because a measurement is usually run from a
            separate thread using the magic command `%%new_job`.
            An error is raised if this has not been satisfied.
            Note that if the measurement is started within a function, no error
            is raised.
        notify: Notify when measurement is complete.
            The function `Measurement.notify_function` must be set


    Notes:
        When the Measurement is started in a separate thread (using %%new_job),
        the Measurement is registered in the user namespace as 'msmt', and the
        dataset as 'data'

    """

    # Context manager
    running_measurement = None
    measurement_thread = None

    # Default names for measurement and dataset, used to set user namespace
    # variables if measurement is executed in a separate thread.
    _default_measurement_name = "msmt"
    _default_dataset_name = "data"
    final_actions = []
    except_actions = []
    max_arrays = 100

    _t_start = None

    # Notification function, called if notify=True.
    # Function should receive the following arguments:
    # Measurement object, exception_type, exception_message, traceback
    # The last three are only not None if an error has occured
    notify_function = None

    def __init__(self, name: str, force_cell_thread: bool = True, notify=False):
        self.name = name

        # Data handler is created during `with Measurement('name')`
        # Used to control dataset(s)
        self.data_handler = None

        # Total dimensionality of loop
        self.loop_shape: Union[Tuple[int], None] = None

        # Current loop indices
        self.loop_indices: Union[Tuple[int], None] = None

        # Index of current action
        self.action_indices: Union[Tuple[int], None] = None

        # contains data groups, such as ParameterNodes and nested measurements
        self._data_groups: Dict[Tuple[int], "MeasurementLoop"] = {}

        # Registry of actions: sweeps, measurements, and data groups
        self.actions: Dict[Tuple[int], Any] = {}
        self.action_names: Dict[Tuple[int], str] = {}

        self.is_context_manager: bool = False  # Whether used as context manager
        self.is_paused: bool = False  # Whether the Measurement is paused
        self.is_stopped: bool = False  # Whether the Measurement is stopped

        self.notify = notify

        self.force_cell_thread = force_cell_thread and using_ipython()

        # Each measurement can have its own final actions, to be executed
        # regardless of whether the measurement finished successfully or not
        # Note that there are also Measurement.final_actions, which are always
        # executed when the outermost measurement finishes
        self.final_actions = []
        self.except_actions = []
        self._masked_properties = []

        self.timings = PerformanceTimer()

    @property
    def dataset(self):
        return self.data_handler.dataset

    def log(self, message: str, level="info"):
        """Send a log message

        Args:
            message: Text to log
            level: Logging level (debug, info, warning, error)
        """
        assert level in ["debug", "info", "warning", "error"]
        logger = logging.getLogger("msmt")
        log_function = getattr(logger, level)

        # Append measurement name
        if self.name is not None:
            message += f" - {self.name}"

        log_function(message)

    @property
    def data_groups(self) -> Dict[Tuple[int], "MeasurementLoop"]:
        if running_measurement() is not None:
            return running_measurement()._data_groups
        else:
            return self._data_groups

    @property
    def active_action(self):
        return self.actions.get(self.action_indices, None)

    @property
    def active_action_name(self):
        return self.action_names.get(self.action_indices, None)

    @property
    def setpoint_list(self):
        if self.data_handler is not None:
            return self.data_handler.setpoint_list
        else:
            return None

    @property
    def measurement_list(self):
        if self.data_handler is not None:
            return self.data_handler.measurement_list
        else:
            return None

    def __enter__(self):
        """Operation when entering a loop"""
        self.is_context_manager = True

        # Encapsulate everything in a try/except to ensure that the context
        # manager is properly exited.
        try:
            if MeasurementLoop.running_measurement is None:
                # Register current measurement as active primary measurement
                MeasurementLoop.running_measurement = self
                MeasurementLoop.measurement_thread = threading.current_thread()

                # Initialize dataset handler
                self.data_handler = DatasetHandler(
                    measurement_loop=self,
                    name=self.name
                )

                # TODO incorporate metadata
                # self._initialize_metadata(self.dataset)
                # with self.timings.record(['dataset', 'save_metadata']):
                #     self.dataset.save_metadata()

                #     if hasattr(self.dataset, 'save_config'):
                #         self.dataset.save_config()

                # Initialize attributes
                self.loop_shape = ()
                self.loop_indices = ()
                self.action_indices = (0,)
                self.data_arrays = {}
                self.set_arrays = {}

                # self.log(f'Measurement started {self.dataset.location}')
                # print(f'Measurement started {self.dataset.location}')

            else:
                if threading.current_thread() is not MeasurementLoop.measurement_thread:
                    raise RuntimeError(
                        "Cannot run a measurement while another measurement "
                        "is already running in a different thread."
                    )

                # Primary measurement is already running. Add this measurement as
                # a data_group of the primary measurement
                msmt = MeasurementLoop.running_measurement
                msmt.data_groups[msmt.action_indices] = self
                data_groups = [
                    (key, getattr(val, 'name', 'None')) for key, val in msmt.data_groups.items()
                ]
                # TODO add metadata
                # msmt.dataset.add_metadata({'data_groups': data_groups})
                msmt.action_indices += (0,)

                # Nested measurement attributes should mimic the primary measurement
                self.loop_shape = msmt.loop_shape
                self.loop_indices = msmt.loop_indices
                self.action_indices = msmt.action_indices
                self.data_arrays = msmt.data_arrays
                self.set_arrays = msmt.set_arrays
                self.timings = msmt.timings

            # Perform measurement thread check, and set user namespace variables
            if self.force_cell_thread and MeasurementLoop.running_measurement is self:
                # Raise an error if force_cell_thread is True and the code is run
                # directly from an IPython cell/prompt but not from a separate thread
                is_main_thread = threading.current_thread() == threading.main_thread()
                if is_main_thread and directly_executed_from_cell():
                    raise RuntimeError(
                        "Measurement must be created in dedicated thread. "
                        "Otherwise specify force_thread=False"
                    )

                # Register the Measurement and data as variables in the user namespace
                # Usually as variable names are 'msmt' and 'data' respectively
                from IPython import get_ipython

                shell = get_ipython()
                shell.user_ns[self._default_measurement_name] = self
                # shell.user_ns[self._default_dataset_name] = self.dataset


            return self
        except:
            # An error has occured, ensure running_measurement is cleared
            if MeasurementLoop.running_measurement is self:
                MeasurementLoop.running_measurement = None
            raise

    def __exit__(self, exc_type: Exception, exc_val, exc_tb):
        """Operation when exiting a loop

        Args:
            exc_type: Type of exception, None if no exception
            exc_val: Exception message, None if no exception
            exc_tb: Exception traceback object, None if no exception
        """
        msmt = MeasurementLoop.running_measurement
        if msmt is self:
            # Immediately unregister measurement as main measurement, in case
            # an error occurs during final actions.
            MeasurementLoop.running_measurement = None

        if exc_type is not None:
            self.log(f"Measurement error {exc_type.__name__}({exc_val})", level="error")

            self._apply_actions(self.except_actions, label="except", clear=True)

            if msmt is self:
                self._apply_actions(
                    MeasurementLoop.except_actions, label="global except", clear=True
                )

        self._apply_actions(self.final_actions, label="final", clear=True)

        self.unmask_all()

        if msmt is self:
            # Also perform global final actions
            # These are always performed when outermost measurement finishes
            self._apply_actions(MeasurementLoop.final_actions, label="global final")

            # Notify that measurement is complete
            if self.notify and self.notify_function is not None:
                try:
                    self.notify_function(exc_type, exc_val, exc_tb)
                except:
                    self.log("Could not notify", level="error")

            t_stop = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # TODO include metadata
            # self.data_handler.add_metadata({"t_stop": t_stop})
            # self.data_handler.add_metadata({"timings": self.timings})
            self.data_handler.finalize()

            self.log(f'Measurement finished')

        else:
            msmt.step_out(reduce_dimension=False)

        self.is_context_manager = False

    # TODO Needs to be implemented
    # def _initialize_metadata(self, dataset):
    #     """Initialize dataset metadata"""
    #     if dataset is None:
    #         dataset = self.dataset

    #     config = qcodes_config
    #     dataset.add_metadata({"config": config})

    #     dataset.add_metadata({"measurement_type": "Measurement"})

    #     # Add instrument information
    #     if Station.default is not None:
    #         dataset.add_metadata({"station": Station.default.snapshot()})

    #     if using_ipython():
    #         measurement_cell = get_last_input_cells(1)[0]

    #         measurement_code = measurement_cell
    #         # If the code is run from a measurement thread, there is some
    #         # initial code that should be stripped
    #         init_string = "get_ipython().run_cell_magic('new_job', '', "
    #         if measurement_code.startswith(init_string):
    #             measurement_code = measurement_code[len(init_string) + 1 : -4]

    #         self._t_start = datetime.now()
    #         dataset.add_metadata(
    #             {
    #                 "measurement_cell": measurement_cell,
    #                 "measurement_code": measurement_code,
    #                 "last_input_cells": get_last_input_cells(20),
    #                 "t_start": self._t_start.strftime('%Y-%m-%d %H:%M:%S')
    #             }
    #         )

    def _verify_action(self, action, name, add_if_new=True):
        """Verify an action corresponds to the current action indices.

        This is only relevant if an action has previously been performed at
        these action indices
        """
        if self.action_indices not in self.actions:
            if add_if_new:
                # Add current action to action registry
                self.actions[self.action_indices] = action
                self.action_names[self.action_indices] = name
        elif name != self.action_names[self.action_indices]:
            raise RuntimeError(
                f"Wrong measurement at action_indices {self.action_indices}. "
                f"Expected: {self.action_names[self.action_indices]}. Received: {name}"
            )

    def _apply_actions(self, actions: list, label="", clear=False):
        """Apply actions, either except_actions or final_actions"""
        for action in actions:
            try:
                action()
            except Exception as e:
                self.log(
                    f"Could not execute {label} action {action} \n"
                    f"{traceback.format_exc()}",
                    level="error",
                )

        if clear:
            actions.clear()

    # Measurement-related functions
    # TODO these methods should always end up with a parameter
    def _measure_parameter(self, parameter, name=None, label=None, unit=None, **kwargs):
        """Measure parameter and store results.

        Called from `measure`.
        MultiParameter is called separately.
        """
        name = name or parameter.name

        # Ensure measuring parameter matches the current action_indices
        self._verify_action(action=parameter, name=name, add_if_new=True)

        # Get parameter result
        result = parameter(**kwargs)

        self.data_handler.add_measurement_result(
            action_indices=self.action_indices,
            result=result,
            parameter=parameter,
            name=name,
            label=label,
            unit=unit,
        )

        return result

    def _measure_multi_parameter(self, multi_parameter, name=None, **kwargs):
        """Measure MultiParameter and store results

        Called from `measure`

        Notes:
            - Does not store setpoints yet
        """
        name = name or multi_parameter.name

        # Ensure measuring multi_parameter matches the current action_indices
        self._verify_action(action=multi_parameter, name=name, add_if_new=True)

        with self.timings.record(['measurement', self.action_indices, 'get']):
            results_list = multi_parameter(**kwargs)

        results = dict(zip(multi_parameter.names, results_list))

        if name is None:
            name = multi_parameter.name

        with MeasurementLoop(name) as msmt:
            for k, (key, val) in enumerate(results.items()):
                msmt.measure(
                    val,
                    name=key,
                    parameter=multi_parameter,
                    label=multi_parameter.labels[k],
                    unit=multi_parameter.units[k],
                )

        return results

    def _measure_callable(self, callable, name=None, **kwargs):
        """Measure a callable (function) and store results

        The function should return a dict, from which each item is measured.
        If the function already contains creates a Measurement, the return
        values aren't stored.
        """
        # Determine name
        if name is None:
            if hasattr(callable, "__self__") and isinstance(
                callable.__self__, InstrumentBase
            ):
                name = callable.__self__.name
            elif hasattr(callable, "__name__"):
                name = callable.__name__
            else:
                action_indices_str = "_".join(str(idx) for idx in self.action_indices)
                name = f"data_group_{action_indices_str}"

        # Ensure measuring callable matches the current action_indices
        self._verify_action(action=callable, name=name, add_if_new=True)

        # Record action_indices before the callable is called
        action_indices = self.action_indices

        results = callable(**kwargs)

        # Check if the callable already performed a nested measurement
        # In this case, the nested measurement is stored as a data_group, and
        # has loop indices corresponding to the current ones.
        msmt = MeasurementLoop.running_measurement
        data_group = msmt.data_groups.get(action_indices)
        if getattr(data_group, "loop_indices", None) != self.loop_indices:
            # No nested measurement has been performed in the callable.
            # Add results, which should be dict, by creating a nested measurement
            if not isinstance(results, dict):
                raise SyntaxError(f"{name} results must be a dict, not {results}")

            with MeasurementLoop(name) as msmt:
                for key, val in results.items():
                    msmt.measure(val, name=key)

        return results

    def _measure_dict(self, value: dict, name: str):
        """Store dictionary results

        Each key is an array name, and the value is the value to store
        """
        if not isinstance(value, dict):
            raise SyntaxError(f"{name} must be a dict, not {value}")

        if not isinstance(name, str) or name == "":
            raise SyntaxError(f"Dict result {name} must have a valid name: {value}")

        # Ensure measuring callable matches the current action_indices
        self._verify_action(action=None, name=name, add_if_new=True)

        with MeasurementLoop(name) as msmt:
            for key, val in value.items():
                msmt.measure(val, name=key)

        return value

    def _measure_value(self, value, name, parameter=None, label=None, unit=None):
        """Store a single value (float/int/bool)

        If this value comes from another parameter acquisition, e.g. from a
        MultiParameter, the parameter can be passed to use the right set arrays.
        """
        if name is None:
            raise RuntimeError("Must provide a name when measuring a value")

        # Ensure measuring callable matches the current action_indices
        self._verify_action(action=None, name=name, add_if_new=True)

        if isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        elif isinstance(value, (bool, np.bool_)):
            value = int(value)

        result = value
        self.data_handler.add_measurement_result(
            action_indices=self.action_indices,
            result=result,
            parameter=parameter,
            name=name,
            label=label,
            unit=unit,
        )
        return result

    def measure(
        self,
        measurable: Union[
            Parameter, Callable, dict, float, int, bool, np.ndarray, None
        ],
        name=None,
        *,  # Everything after here must be a kwarg
        label=None,
        unit=None,
        timestamp=False,
        **kwargs,
    ):
        """Perform a single measurement of a Parameter, function, etc.


        Args:
            measurable: Item to measure. Can be one of the following:
                Parameter
                Callable function/method, which should either perform a nested
                    Measurement, or return a dict.
                    In the case of returning a dict, all the key/value pairs
                    are grouped together.
                float, int, bool, array
            name: Optional name for measured element or data group.
                If the measurable is a float, int, bool, or array, the name is
                mandatory.
                Otherwise, the default name is used.
            label: Optional label, is ignored if measurable is a Parameter or callable
            unit: Optional unit, is ignored if measurable is a Parameter or callable
            timestamp: If True, the timestamps immediately before and after this
                       measurement are recorded

        Returns:
            Return value of measurable
        """
        if not self.is_context_manager:
            raise RuntimeError(
                "Must use the Measurement as a context manager, "
                "i.e. 'with Measurement(name) as msmt:'"
            )
        elif self.is_stopped:
            raise SystemExit("Measurement.stop() has been called")
        elif threading.current_thread() is not MeasurementLoop.measurement_thread:
            raise RuntimeError(
                "Cannot measure while another measurement is already running "
                "in a different thread."
            )

        if self != MeasurementLoop.running_measurement:
            # Since this Measurement is not the running measurement, it is a
            # DataGroup in the running measurement. Delegate measurement to the
            # running measurement
            return MeasurementLoop.running_measurement.measure(
                measurable, name=name, label=label, unit=unit, **kwargs
            )

        # Code from hereon is only reached by the primary measurement,
        # i.e. the running_measurement

        # Wait as long as the measurement is paused
        while self.is_paused:
            sleep(0.1)

        t0 = perf_counter()
        initial_action_indices = self.action_indices

        if timestamp:
            t_now = datetime.now()

            # Store time referenced to t_start
            self.measure((t_now - self._t_start).total_seconds(),
                         'T_pre', unit='s', timestamp=False)
            self.skip()  # Increment last action index by 1

        # TODO Incorporate kwargs name, label, and unit, into each of these
        if isinstance(measurable, Parameter):
            result = self._measure_parameter(
                measurable, name=name, label=label, unit=unit, **kwargs
            )
            self.skip()  # Increment last action index by 1
        elif isinstance(measurable, MultiParameter):
            result = self._measure_multi_parameter(measurable, name=name, **kwargs)
        elif callable(measurable):
            result = self._measure_callable(measurable, name=name, **kwargs)
        elif isinstance(measurable, dict):
            result = self._measure_dict(measurable, name=name)
        elif isinstance(measurable, RAW_VALUE_TYPES):
            result = self._measure_value(measurable, name=name, label=label, unit=unit, **kwargs)
            self.skip()  # Increment last action index by 1
        else:
            raise RuntimeError(
                f"Cannot measure {measurable} as it cannot be called, and it "
                f"is not a dict, int, float, bool, or numpy array."
            )

        if timestamp:
            t_now = datetime.now()

            # Store time referenced to t_start
            self.measure((t_now - self._t_start).total_seconds(),
                         'T_post', unit='s', timestamp=False)
            self.skip()  # Increment last action index by 1


        self.timings.record(
            ['measurement', initial_action_indices, 'total'],
            perf_counter() - t0
        )

        return result

    # Methods related to masking of parameters/attributes/keys
    def _mask_attr(self, obj: object, attr: str, value):
        """Temporarily override an object attribute during the measurement.

        The value will be reset at the end of the measurement
        This can also be a nested measurement.

        Args:
            obj: Object whose value should be masked
            attr: Attribute to be masked
            val: Masked value

        Returns:
            original value
        """
        original_value = getattr(obj, attr)
        setattr(obj, attr, value)

        self._masked_properties.append(
            {
                "type": "attr",
                "obj": obj,
                "attr": attr,
                "original_value": original_value,
                "value": value,
            }
        )

        return original_value

    def _mask_parameter(self, param, value):
        """Temporarily override a parameter value during the measurement.

        The value will be reset at the end of the measurement.
        This can also be a nested measurement.

        Args:
            param: Parameter whose value should be masked
            val: Masked value

        Returns:
            original value
        """
        original_value = param()
        param(value)

        self._masked_properties.append(
            {
                "type": "parameter",
                "obj": param,
                "original_value": original_value,
                "value": value,
            }
        )

        return original_value

    def _mask_key(self, obj: dict, key: str, value):
        """Temporarily override a dictionary key during the measurement.

        The value will be reset at the end of the measurement
        This can also be a nested measurement.

        Args:
            obj: dictionary whose value should be masked
            key: key to be masked
            val: Masked value

        Returns:
            original value
        """
        original_value = obj[key]
        obj[key] = value

        self._masked_properties.append(
            {
                "type": "key",
                "obj": obj,
                "key": key,
                "original_value": original_value,
                "value": value,
            }
        )

        return original_value

    def mask(self, obj: Union[object, dict], val=None, **kwargs):
        """Mask a key/attribute/parameter for the duration of the Measurement

        Multiple properties can be masked by passing as kwargs.
        Masked properties are reverted at the end of the measurement, even if
        the measurement crashes

        Args:
            obj: Object from which to mask property.
                For a dict, an item is masked.
                For a ParameterNode, a parameter is masked.
                For a parameter, the value is masked.
                For all other objects, an attribute is masked.
            val: Masked value, only relevant if obj is a parameter
            **kwargs: Masked properties

        Returns:
            List of original values before masking

        Examples:
            ```
            node = ParameterNode()
            node.p1 = Parameter(initial_value=1, set_cmd=None)

            with Measurement('test_masking') as msmt:
                msmt.mask(node, p1=2)
                print(f"node.p1 has value {node.p1}")
            >>> node.p1 has value 2
            print(f"node.p1 has value {node.p1}")
            >>> node.p1 has value 1
            ```
        """
        if isinstance(obj, InstrumentBase):
            assert val is None
            # kwargs can be either parameters or attrs
            return [
                self._mask_parameter(obj.parameters[key], val)
                if key in obj.parameters
                else self._mask_attr(obj, key, val)
                for key, val in kwargs.items()
            ]
        if isinstance(obj, Parameter) and not kwargs:
            # if kwargs are passed, they are to be treated as attrs
            return self._mask_parameter(obj, val)
        elif isinstance(obj, dict):
            if not kwargs:
                raise SyntaxError("Must pass kwargs when masking a dict")
            return [self._mask_key(obj, key, val) for key, val in kwargs.items()]
        else:
            if not kwargs:
                raise SyntaxError("Must pass kwargs when masking")
            return [self._mask_attr(obj, key, val) for key, val in kwargs.items()]

    def unmask(
        self,
        obj,
        attr=None,
        key=None,
        type=None,
        value=None,
        raise_exception=True,
        **kwargs  # Add kwargs because original_value may be None
    ):
        if 'original_value' not in kwargs:
            # No masked property passed. We collect all the masked properties
            # that satisfy these requirements and unmask each of them.
            unmask_properties = []
            remaining_masked_properties = []
            for masked_property in self._masked_properties:
                if masked_property["obj"] != obj:
                    remaining_masked_properties.append(masked_property)
                elif attr is not None and masked_property.get("attr") != attr:
                    remaining_masked_properties.append(masked_property)
                elif key is not None and masked_property.get("key") != key:
                    remaining_masked_properties.append(masked_property)
                else:
                    unmask_properties.append(masked_property)

            for unmask_property in reversed(unmask_properties):
                self.unmask(**unmask_property)

            self._masked_properties = remaining_masked_properties
        else:
            # A masked property has been passed, which we unmask here
            try:
                original_value = kwargs['original_value']
                if type == "key":
                    obj[key] = original_value
                elif type == "attr":
                    setattr(obj, attr, original_value)
                elif type == "parameter":
                    obj(original_value)
                else:
                    raise SyntaxError(f"Unmask type {type} not understood")
            except Exception as e:
                self.log(
                    f"Could not unmask {obj} {type} from masked value {value} "
                    f"to original value {original_value}\n"
                    f"{traceback.format_exc()}",
                    level="error",
                )

                if raise_exception:
                    raise e

    def unmask_all(self):
        """Unmask all masked properties"""
        masked_properties = reversed(self._masked_properties)
        for masked_property in masked_properties:
            self.unmask(**masked_property, raise_exception=False)
        self._masked_properties.clear()

    # Functions relating to measurement flow
    def pause(self):
        """Pause measurement at start of next parameter sweep/measurement"""
        running_measurement().is_paused = True

    def resume(self):
        """Resume measurement after being paused"""
        running_measurement().is_paused = False

    def stop(self):
        """Stop measurement at start of next parameter sweep/measurement"""
        running_measurement().is_stopped = True
        # Unpause loop
        running_measurement().resume()

    def skip(self, N=1):
        """Skip an action index.

        Useful if a measure is only sometimes run

        Args:
            N: number of action indices to skip

        Examples:
            This measurement repeatedly creates a random value.
            It then stores the value twice, but the first time the value is
            only stored if it is above a threshold. Notice that if the random
            value is not above this threshold, the second measurement would
            become the first measurement if msmt.skip is not called
            ```
            with Measurement('skip_measurement') as msmt:
                for k in Sweep(range(10)):
                    random_value = np.random.rand()
                    if random_value > 0.7:
                        msmt.measure(random_value, 'random_value_conditional')
                    else:
                        msmt.skip()

                    msmt.measure(random_value, 'random_value_unconditional)
            ```
        """
        if running_measurement() is not self:
            return running_measurement().skip(N=N)
        else:
            action_indices = list(self.action_indices)
            action_indices[-1] += N
            self.action_indices = tuple(action_indices)
            return self.action_indices

    def step_out(self, reduce_dimension=True):
        """Step out of a Sweep

        This function usually doesn't need to be called.
        """
        if MeasurementLoop.running_measurement is not self:
            MeasurementLoop.running_measurement.step_out(reduce_dimension=reduce_dimension)
        else:
            if reduce_dimension:
                self.loop_shape = self.loop_shape[:-1]
                self.loop_indices = self.loop_indices[:-1]

            # Remove last action index and increment one before that by one
            action_indices = list(self.action_indices[:-1])
            action_indices[-1] += 1
            self.action_indices = tuple(action_indices)

    def traceback(self):
        """Print traceback if an error occurred.

         Measurement must be ran from separate thread
        """
        if self.measurement_thread is None:
            raise RuntimeError('Measurement was not started in separate thread')
        else:
            self.measurement_thread.traceback()


def running_measurement() -> MeasurementLoop:
    """Return the running measurement"""
    return MeasurementLoop.running_measurement


class _IterateDondSweep:
    def __init__(self, sweep: AbstractSweep):
        self.sweep = sweep
        self.iterator = None
        self.parameter = sweep._param

    def __len__(self):
        return self.sweep.num_points

    def __iter__(self):
        self.iterator = iter(self.sweep.get_setpoints())
        return self

    def __next__(self):
        value = next(self.iterator)
        self.sweep._param(value)

        for action in self.sweep.post_actions:
            action()

        if self.sweep.delay:
            sleep(self.sweep.delay)

        return value



class BaseSweep(AbstractSweep):
    """Sweep over an iterable inside a Measurement

    Args:
        sequence: Sequence to iterate over.
            Can be an iterable, or a parameter Sweep.
            If the sequence
        name: Name of sweep. Not needed if a Parameter is passed
        unit: unit of sweep. Not needed if a Parameter is passed
        revert: Stores the state of a parameter before sweeping it,
            then reverts the original value upon exiting the loop.
        delay: Wait time after setting value (default zero).

    Examples:
        ```
        with Measurement('sweep_msmt') as msmt:
            for value in Sweep(np.linspace(5), 'sweep_values'):
                msmt.measure(value, 'linearly_increasing_value')

            p = Parameter('my_parameter')
            for param_val in Sweep(p.
        ```
    """
    def __init__(self, sequence, name=None, label=None, unit=None, parameter=None, revert=False, delay=None, initial_delay=None):
        if isinstance(sequence, AbstractSweep):
            sequence = _IterateDondSweep(sequence)
        elif not isinstance(sequence, Iterable):
            raise SyntaxError(f"Sweep sequence must be iterable, not {type(sequence)}")

        # Properties for the data array
        self.name = name
        self.label = label
        self.unit = unit
        self.parameter = parameter

        self.sequence = sequence
        self.dimension = None
        self.loop_index = None
        self.iterator = None
        self.revert = revert
        self._delay = delay
        self.initial_delay = initial_delay

        # setpoint_info will be populated once the sweep starts
        self.setpoint_info = None

        # Validate values
        if self.parameter is not None and hasattr(self.parameter, 'validate'):
            for value in self.sequence:
                self.parameter.validate(value)

    def __repr__(self):
        components = []

        # Add parameter or name
        if self.parameter is not None:
            components.append(f'parameter={self.parameter}')
        elif self.name is not None:
            components.append(f"'{self.name}'")

        # Add number of elements
        num_elems = str(len(self.sequence)) if self.sequence is not None else 'unknown'
        components.append(f'length={num_elems}')

        # Combine components
        components_str = ', '.join(components)
        return f'Sweep({components_str})'

    def __iter__(self):
        if threading.current_thread() is not MeasurementLoop.measurement_thread:
            raise RuntimeError(
                "Cannot create a Sweep while another measurement "
                "is already running in a different thread."
            )

        msmt = running_measurement()
        if msmt is None:
            raise RuntimeError("Cannot start a sweep outside a Measurement")

        if self.revert:
            if isinstance(self.sequence, SweepValues):
                msmt.mask(self.sequence.parameter, self.sequence.parameter.get())
            else:
                raise NotImplementedError("Unable to revert non-parameter values.")

        self.loop_index = 0
        self.dimension = len(msmt.loop_shape)
        self.iterator = iter(self.sequence)

        # Create setpoint_list
        self.setpoint_info = self.initialize()

        msmt.loop_shape += (len(self.sequence),)
        msmt.loop_indices += (self.loop_index,)
        msmt.action_indices += (0,)

        return self

    def __next__(self):
        msmt = running_measurement()

        if not msmt.is_context_manager:
            raise RuntimeError(
                "Must use the Measurement as a context manager, "
                "i.e. 'with Measurement(name) as msmt:'"
            )
        elif msmt.is_stopped:
            raise SystemExit

        # Wait as long as the measurement is paused
        while msmt.is_paused:
            sleep(0.1)

        # Increment loop index of current dimension
        loop_indices = list(msmt.loop_indices)
        loop_indices[self.dimension] = self.loop_index
        msmt.loop_indices = tuple(loop_indices)

        try:  # Perform loop action
            sweep_value = next(self.iterator)
            # Remove last action index and increment one before that by one
            action_indices = list(msmt.action_indices)
            action_indices[-1] = 0
            msmt.action_indices = tuple(action_indices)
        except StopIteration:  # Reached end of iteration
            if self.revert:
                if isinstance(self.sequence, SweepValues):
                    msmt.unmask(self.sequence.parameter)
                else:
                    # TODO: Check what other iterators might be able to be masked
                    pass
            self.exit_sweep()

        # Set parameter if passed along
        if self.parameter is not None and self.parameter.settable:
            self.parameter(sweep_value)

        # Optional wait after settings value
        if self.initial_delay and self.loop_index == 0:
            sleep(self.initial_delay)
        if self.delay:
            sleep(self.delay)

        self.setpoint_info['latest_value'] = sweep_value

        self.loop_index += 1

        return sweep_value

    def initialize(self):
        msmt = running_measurement()
        if msmt.action_indices in msmt.setpoint_list:
            return msmt.setpoint_list[msmt.action_indices]

        # Determine sweep parameter
        if self.parameter is None:
            if isinstance(self.sequence, _IterateDondSweep):
                # sweep is a doNd sweep that already has a parameter
                self.parameter = self.sequence.parameter
            else:
                # Need to create a parameter
                self.parameter = Parameter(
                    name=self.name,
                    label=self.label,
                    unit=self.unit
                )

        setpoint_info = {
            'parameter': self.parameter,
            'latest_value': None,
            'registered': False
        }

        # Add to setpoint list
        msmt.setpoint_list[msmt.action_indices] = setpoint_info

        # Add to measurement actions
        assert msmt.action_indices not in msmt.actions
        msmt.actions[msmt.action_indices] = self

        return setpoint_info

    def exit_sweep(self):
        msmt = running_measurement()
        msmt.step_out(reduce_dimension=True)
        raise StopIteration

    def execute(
        self,
        *args: Iterable['BaseSweep'],
        name: str = None,
        measure_params: Iterable = None,
        repetitions: int = 1,
        sweep: Union[Iterable, 'BaseSweep'] = None
    ):
        # Get "measure_params" from station if not provided
        if measure_params is None:
            station = Station.default
            if station is None or not getattr(station, 'measure_params', None):
                raise RuntimeError(
                    'Cannot determine parameters to measure. '
                    'Either provide measure_params, or set station.measure_params'
                )
            measure_params = station.measure_params

        # Create list of sweeps
        sweeps = list(args)
        if not all(isinstance(sweep, BaseSweep) for sweep in sweeps):
            raise ValueError('Args passed to Sweep.execute must be Sweeps')
        if isinstance(sweep, BaseSweep):
            sweeps.append(sweep)
        elif isinstance(sweep, (list, tuple)):
            sweeps += list(sweep)

        # Add repetition as a sweep if > 1
        if repetitions > 1:
            repetition_sweep = BaseSweep(range(repetitions), name='repetition')
            sweeps = [repetition_sweep] + sweeps

        # Add self as innermost sweep
        sweeps += [self]

        # Determine "name" if not provided from sweeps
        if name is None:
            dimensionality = 1 + len(sweeps)
            sweep_names = [str(sweep.name) for sweep in sweeps] + [str(self.name)]
            name = f'{dimensionality}D_sweep_' + '_'.join(sweep_names)

        with MeasurementLoop(name) as msmt:
            measure_sweeps(sweeps=sweeps, measure_params=measure_params, msmt=msmt)

    # Methods needed to make BaseSweep subclass of AbstractSweep
    def get_setpoints(self) -> np.ndarray:
        return self.sequence

    @property
    def param(self) -> ParameterBase:
        # TODO create necessary parameter if self.parameter is None
        return self.parameter

    @property
    def num_points(self) -> float:
        return len(self.sequence)

    @property
    def delay(self) -> float:
        """
        Delay between two consecutive sweep points.
        """
        return self._delay or 0

    @property
    def post_actions(self) -> ActionsT:
        # TODO maybe add option for post actions
        # However this can cause issues if sweep is prematurely exited
        return []




class Sweep(BaseSweep):
    sequence_keywords = ['start', 'stop', 'around', 'num', 'step', 'parameter', 'sequence']
    base_keywords = ['delay', 'initial_delay', 'name', 'label', 'unit', 'revert', 'parameter']

    def __init__(
        self,
        *args,
        start: float = None,
        stop: float = None,
        around: float = None,
        num: int = None,
        step: float = None,
        delay: float = None,
        initial_delay: float = None,
        name: str = None,
        label: str = None,
        unit: str = None,
        revert: bool = None
    ):
        kwargs = dict(
            start=start,
            stop=stop,
            around=around,
            num=num,
            step=step,
            delay=delay,
            initial_delay=initial_delay,
            name=name,
            label=label,
            unit=unit,
            revert=revert
        )

        sequence_kwargs, base_kwargs = self._transform_args_to_kwargs(*args, **kwargs)

        self._explicit_sequence = None
        self.sequence = self._generate_sequence(**sequence_kwargs)

        super().__init__(sequence=self.sequence, **base_kwargs)

    def _transform_args_to_kwargs(self, *args, **kwargs):
        """Transforms sweep initialization args to kwargs.
        Allowed args are:

        1 arg:
        - Sweep([1,2,3], name='name')
          : sweep over sequence [1,2,3] with sweep array name 'name'
          Note that kwarg "name" must be provided
        - Sweep(parameter, stop=stop_val)
          : sweep "parameter" from current value to "stop_val"
        - Sweep(parameter, around=around_val)
          : sweep "parameter" around current value with range "around_val"
        2 args:
        - Sweep(parameter, [1,2,3])
          : sweep "parameter" over sequence [1,2,3]
        - Sweep(parameter, stop_val)
          : sweep "parameter" from current value to "stop_val"
        - Sweep([1,2,3], 'name')
          : sweep over sequence [1,2,3] with sweep array name 'name'
        3 args:
        - Sweep(parameter, start_val, stop_val)
          : sweep "parameter" from "start_val" to "stop_val"
          If "num" or "step" is not given as kwarg, it will check if "num" or "step"
          if set in dict "parameter.sweep_defaults" and use that, or raise an error otherwise.
        4 args:
        - Sweep(parameter, start_val, stop_val, num)
          : Sweep "parameter" from "start_val" to "stop_val" with "num" number of points
        """
        if len(args) == 1:  # Sweep([1,2,3], name='name')
            if isinstance(args[0], Iterable):
                assert kwargs.get('name') is not None, "Must provide name if sweeping iterable"
                kwargs['sequence'], = args
            elif isinstance(args[0], _BaseParameter):
                assert kwargs.get('stop') is not None or kwargs.get('around') is not None, \
                    "Must provide stop value for parameter"
                kwargs['parameter'], = args
            else:
                raise SyntaxError('Sweep with 1 arg must have iterable or parameter as arg')
        elif len(args) == 2:
            if isinstance(args[0], _BaseParameter):  # Sweep(parameter, [1,2,3])
                if isinstance(args[1], Iterable):
                    kwargs['parameter'], kwargs['sequence'] = args
                elif isinstance(args[1], (int, float)):
                    kwargs['parameter'], kwargs['stop'] = args
                else:
                    raise SyntaxError('Sweep with Parameter arg and second arg should h')
            elif isinstance(args[0], Iterable):  # Sweep([1,2,3], 'name')
                assert isinstance(args[1], str)
                assert kwargs.get('name') is None
                kwargs['sequence'], kwargs['name'] = args
            else:
                raise SyntaxError(
                    'Unknown sweep syntax. Either use "Sweep(parameter, sequence)" or '
                    'Sweep(sequence, name)"'
                )
        elif len(args) == 3:  # Sweep(parameter, 0, 1)
            assert isinstance(args[0], _BaseParameter)
            assert isinstance(args[1], (float, int))
            assert isinstance(args[2], (float, int))
            assert kwargs.get('start') is None
            assert kwargs.get('stop') is None
            kwargs['parameter'], kwargs['start'], kwargs['stop'] = args
        elif len(args) == 4:  # Sweep(parameter, 0, 1, 151)
            assert isinstance(args[0], _BaseParameter)
            assert isinstance(args[1], (float, int))
            assert isinstance(args[2], (float, int))
            assert isinstance(args[3], (float, int))
            assert kwargs.get('start') is None
            assert kwargs.get('stop') is None
            assert kwargs.get('num') is None
            kwargs['parameter'], kwargs['start'], kwargs['stop'], kwargs['num'] = args

        # Use parameter name, label, and unit if not explicitly provided
        if kwargs.get('parameter') is not None:
            kwargs.setdefault('name', kwargs['parameter'].name)
            kwargs.setdefault('label', kwargs['parameter'].label)
            kwargs.setdefault('unit', kwargs['parameter'].unit)

            # Update kwargs with sweep_defaults from parameter
            if hasattr(kwargs['parameter'], 'sweep_defaults'):
                for key, val in kwargs['parameter'].sweep_defaults.items():
                    if kwargs.get(key) is None:
                        kwargs[key] = val

        sequence_kwargs = {key: kwargs.get(key) for key in self.sequence_keywords}
        base_kwargs = {key: kwargs.get(key) for key in self.base_keywords}

        return sequence_kwargs, base_kwargs

    def _generate_sequence(self, start=None, stop=None, around=None, num=None, step=None, parameter=None, sequence=None):
        """Creates a sequence from passed values"""
        # Return "sequence" if explicitly provided
        if sequence is not None:
            return sequence

        # Verify that "around" is used with "parameter" but not with "start" and "stop"
        if around is not None:
            if start is not None or stop is not None:
                raise SyntaxError('Cannot pass kwarg "around" and also "start" or "stop')
            elif parameter is None:
                raise SyntaxError('Cannot use kwarg "around" without a parameter')

            # Convert "around" to "start" and "stop" using parameter current value
            center_value = parameter()
            if center_value is None:
                raise ValueError('Parameter must have initial value if "around" keyword is used')
            start = center_value - around
            stop = center_value + around
        elif stop is not None:
            # Use "parameter" current value if "start" is not provided
            if start is None:
                if parameter is None:
                    raise SyntaxError('Cannot use "stop" without "start" or a "parameter"')
                start = parameter()
                if start is None:
                    raise ValueError('Parameter must have initial value if start is not explicitly provided')
        else:
            raise SyntaxError('Must provide either "around" or "stop"')

        if num is not None:
            sequence = np.linspace(start, stop, num)
        elif step is not None:
            # Ensure step is positive
            step = abs(step) if stop > start else -abs(step)

            sequence = np.arange(start, stop, step)

            # Append final datapoint
            if abs((stop - sequence[-1]) / step) > 1e-9:
                sequence = np.append(sequence, [stop])
        else:
            raise SyntaxError('Cannot determine measurement points. Either provide "sequence, "step" or "num"')

        return sequence


class RepetitionSweep(BaseSweep):
    def __init__(self, repetitions, start=0, name='repetition', label='Repetition', unit=None):
        self.start = start
        self.repetitions = repetitions
        sequence = start + np.arange(repetitions)

        super().__init__(sequence, name, label, unit)


def measure_sweeps(sweeps: list[BaseSweep], measure_params: list[_BaseParameter], msmt: MeasurementLoop = None):
    """Recursively iterate over Sweep objects, measuring measure_params in innermost loop

    Args:
        sweeps: list of BaseSweep objects to sweep over
        measure_params: list of parameters to measure in innermost loop
    """
    if sweeps:
        outer_sweep, *inner_sweeps = sweeps

        for _ in outer_sweep:
            measure_sweeps(inner_sweeps, measure_params, msmt=msmt)

    else:
        if msmt is None:
            msmt = running_measurement()

        for measure_param in measure_params:
            msmt.measure(measure_param)
