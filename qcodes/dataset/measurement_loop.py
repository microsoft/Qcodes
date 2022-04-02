from enum import unique
import numpy as np
from collections import Counter
from typing import List, Tuple, Union, Sequence, Dict, Any, Callable, Iterable
import threading
from time import sleep, perf_counter
import traceback
import logging
from datetime import datetime

from qcodes.dataset.measurements import Measurement
from qcodes.dataset.descriptions.detect_shapes import detect_shape_of_measurement
from qcodes.station import Station
from qcodes.instrument.base import InstrumentBase
from qcodes.instrument.sweep_values import SweepValues
from qcodes.instrument.parameter import DelegateParameter, Parameter, MultiParameter
from qcodes.utils.helpers import (
    using_ipython,
    directly_executed_from_cell,
    get_last_input_cells,
    PerformanceTimer
)
from qcodes import config as qcodes_config

RAW_VALUE_TYPES = (float, int, bool, np.ndarray, np.integer, np.floating, np.bool_, type(None))


class DatasetHandler:
    """Handler for a single DataSet (with Measurement and Runner)"""
    def __init__(self):
        self.initialized = False
        self.dataset = None
        self.runner = None
        self.measurement = None

        # Key: action_index
        # Values:
        # - parameter
        # - dataset_parameter (differs from 'parameter' when multiple share same name)
        self.setpoint_list = dict()

        self.measurement_list = dict()
        # Dict with key being action_index and value is a dict containing
        # - parameter
        # - setpoint_parameters
        # - shape
        # - unstored_results - list where each element contains (*setpoints, measurement_value)

    def initialize(self):
        # Once initialized, no new parameters can be added
        assert not self.initialized, "Cannot initialize twice"

        self.measurement = Measurement()

        # Register all setpoints parameters
        self._create_unique_dataset_parameters(self.setpoint_list)
        for setpoint_info in self.setpoint_list.values():
            self.measurement.register_parameter(setpoint_info['dataset_parameter'])
            
        # Register all measurement parameters
        self._create_unique_dataset_parameters(self.measurement_list)
        for measurement_info in self.measurement_list.values():
            self.measurement.register_parameter(
                measurement_info['dataset_parameter'],
                setpoints=measurement_info['setpoint_parameters']
            )
            self.measurement.set_shapes(
                detect_shape_of_measurement(
                    (measurement_info['dataset_parameter'],), 
                    measurement_info['shape']
                )
            )

        # Create measurement Runner
        self.runner = self.measurement.run()

        # Create measurement Dataset
        self.dataset = self.runner.__enter__()

        # Add results that were taken before initializing dataset
        for measurement_info in self.measurement_list.values():
            for unstored_result in measurement_info['unstored_results']:
                parameters = *measurement_info['setpoint_parameters'], measurement_info['dataset_parameter']
                result = tuple(zip(parameters, unstored_result))
                self.dataset.add_result(result)

        self.initialized = True

    def _create_unique_dataset_parameters(self, parameter_list):
        """Populates 'dataset_parameter' of parameter_list
        
        Ensure parameters have unique names
        """
        parameter_names = [param_info['parameter'].name for param_info in parameter_list]
        duplicate_names = [name for name, count in Counter(parameter_names) if count > 1]
        unique_names = [name for name, count in Counter(parameter_names) if count == 1]

        for name in unique_names:
            parameter_info = next(
                param_info for param_info in parameter_list 
                if param_info['parameter'].name == name
            )
            parameter_info['dataset_parameter'] = parameter_info['parameter']

        for name in duplicate_names:
            # Need to rename parameters with duplicate names
            duplicate_parameter_info_list = [
                param_info for param_info in parameter_list 
                if param_info['parameter'].name == name
            ]
            for k, parameter_info in duplicate_parameter_info_list:
                # Create delegate parameter
                delegate_parameter = DelegateParameter(
                    name=f"{parameter_info['parameter'].name}_{k}",
                    source=parameter_info['parameter']
                )
                parameter_info['dataset_parameter'] = delegate_parameter


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
        # TODO Need to handle when a parameter is not being passed
        if action_indices not in self.measurement_list:
            assert not self.initialized, "Cannot measure parameter for the first time after initializing dataset"

            self.measurement_list[action_indices] = {
                'parameter': parameter,
                'setpoint_parameters': None,  # TODO
                'shape': None, # TODO
                'unstored_results': []
            }

        # Select existing array
        data_array = self.data_arrays[action_indices]

        # Ensure an existing data array has the correct name
        # parameter can also be a string, in which case we don't use parameter.name
        if name is None:
            name = parameter.name

        # TODO is this the right place for this check?
        if not data_array.name == name:
            raise SyntaxError(
                f"Existing DataArray '{data_array.name}' differs from result {name}"
            )

        data_to_store = {data_array.array_id: result}

        # If result is an array, update set_array elements
        if isinstance(result, list):  # Convert result list to array
            result = np.ndarray(result)
        if isinstance(result, np.ndarray):
            ndim = len(self.loop_indices)
            if len(data_array.set_arrays) != ndim + result.ndim:
                raise RuntimeError(
                    f"Wrong number of set arrays for {data_array.name}. "
                    f"Expected {ndim + result.ndim} instead of "
                    f"{len(data_array.set_arrays)}."
                )

            for k, set_array in enumerate(data_array.set_arrays[ndim:]):
                # Successive set arrays must increase dimensionality by unity
                arr = np.arange(result.shape[k])
                if parameter is not None and hasattr(parameter, 'setpoints') \
                        and parameter.setpoints is not None:
                    arr_idx = parameter.names.index(name)
                    arr = parameter.setpoints[arr_idx][k]

                # Add singleton dimensions
                arr = np.broadcast_to(arr, result.shape[: k + 1])
                data_to_store[set_array.array_id] = arr

        # Use dummy index if there are no loop indices.
        # This happens if the measurement is performed outside a Sweep
        loop_indices = self.loop_indices
        if not loop_indices and not isinstance(result, (list, np.ndarray)):
            loop_indices = (0,)

        return data_to_store
    

class DataHandler:
    def __init__(self, measurement_loop):
        # MeasurementLoop corresponding to this DataHandler
        # Cannot be a nested MeasurementLoop
        self.measurement_loop = measurement_loop

        self.dataset_handlers = []

    @property
    def active_dataset_handler(self):
        # TODO Allow for multiple possible measurements
        return self.measurements[0]

    def finalize(self):
        """Called when outermost measurement is finished"""

    def create_dataset(self):
        pass

    def new_dataset(self):
        pass

    def add_metadata(self):
        pass

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
        if action_indices not in self.data_arrays:
            # Create array based on first result type and shape
            self._create_data_array(
                action_indices,
                result,
                parameter=parameter,
                name=name,
                label=label,
                unit=unit,
            )

        # Select existing array
        data_array = self.data_arrays[action_indices]

        # Ensure an existing data array has the correct name
        # parameter can also be a string, in which case we don't use parameter.name
        if name is None:
            name = parameter.name

        # TODO is this the right place for this check?
        if not data_array.name == name:
            raise SyntaxError(
                f"Existing DataArray '{data_array.name}' differs from result {name}"
            )

        data_to_store = {data_array.array_id: result}

        # If result is an array, update set_array elements
        if isinstance(result, list):  # Convert result list to array
            result = np.ndarray(result)
        if isinstance(result, np.ndarray):
            ndim = len(self.loop_indices)
            if len(data_array.set_arrays) != ndim + result.ndim:
                raise RuntimeError(
                    f"Wrong number of set arrays for {data_array.name}. "
                    f"Expected {ndim + result.ndim} instead of "
                    f"{len(data_array.set_arrays)}."
                )

            for k, set_array in enumerate(data_array.set_arrays[ndim:]):
                # Successive set arrays must increase dimensionality by unity
                arr = np.arange(result.shape[k])
                if parameter is not None and hasattr(parameter, 'setpoints') \
                        and parameter.setpoints is not None:
                    arr_idx = parameter.names.index(name)
                    arr = parameter.setpoints[arr_idx][k]

                # Add singleton dimensions
                arr = np.broadcast_to(arr, result.shape[: k + 1])
                data_to_store[set_array.array_id] = arr

        # Use dummy index if there are no loop indices.
        # This happens if the measurement is performed outside a Sweep
        loop_indices = self.loop_indices
        if not loop_indices and not isinstance(result, (list, np.ndarray)):
            loop_indices = (0,)

        return data_to_store

    # Data array functions
    # TODO Needs to be reformed
    def _create_data_array(
        self,
        action_indices: Tuple[int],
        result,
        parameter: Parameter = None,
        is_setpoint: bool = False,
        name: str = None,
        label: str = None,
        unit: str = None,
    ):
        """Create a data array from a parameter and result.

        The data array shape is extracted from the result shape, and the current
        loop dimensions.

        The data array is added to the current data set.

        Args:
            parameter: Parameter for which to create a DataArray. Can also be a
                string, in which case it is the data_array name
            result: Result returned by the Parameter
            action_indices: Action indices for which to store parameter
            is_setpoint: Whether the Parameter is used for sweeping or measuring
            label: Data array label. If not provided, the parameter label is
                used. If the parameter is a name string, the label is extracted
                from the name.
            unit: Data array unit. If not provided, the parameter unit is used.

        Returns:
            Newly created data array

        """
        if parameter is None and name is None:
            raise SyntaxError(
                "When creating a data array, must provide either a parameter or a name"
            )

        if len(running_measurement().data_arrays) >= self.max_arrays:
            raise RuntimeError(
                f"Number of arrays in dataset exceeds "
                f"Measurement.max_arrays={self.max_arrays}. Perhaps you forgot"
                f"to encapsulate a loop with a Sweep()?"
            )

        array_kwargs = {
            "is_setpoint": is_setpoint,
            "action_indices": action_indices,
            "shape": self.loop_shape,
        }

        if is_setpoint or isinstance(result, (np.ndarray, list)):
            array_kwargs["shape"] += np.shape(result)

        # Use dummy index (1, ) if measurement is performed outside a Sweep
        if not array_kwargs["shape"]:
            array_kwargs["shape"] = (1,)

        if isinstance(parameter, Parameter):
            array_kwargs["parameter"] = parameter
            # Add a custom name
            if name is not None:
                array_kwargs["full_name"] = name
            if label is not None:
                array_kwargs["label"] = label
            if unit is not None:
                array_kwargs["unit"] = unit
        else:
            array_kwargs["name"] = name
            if label is None:
                label = name[0].capitalize() + name[1:].replace("_", " ")
            array_kwargs["label"] = label
            array_kwargs["unit"] = unit or ""

        # Add setpoint arrays
        if not is_setpoint:
            array_kwargs["set_arrays"] = self._add_set_arrays(
                action_indices, result, parameter=parameter, name=(name or parameter.name)
            )

        data_array = DataArray(**array_kwargs)

        data_array.array_id = data_array.full_name
        data_array.array_id += "_" + "_".join(str(k) for k in action_indices)

        data_array.init_data()

        self.dataset.add_array(data_array)
        with self.timings.record(['dataset', 'save_metadata']):
            self.dataset.save_metadata()

        # Add array to set_arrays or to data_arrays of this Measurement
        if is_setpoint:
            self.set_arrays[action_indices] = data_array
        else:
            self.data_arrays[action_indices] = data_array

        return data_array

    def _add_set_arrays(
        self, action_indices: Tuple[int], result, name: str, parameter: Union[Parameter, None] = None
    ):
        """Create set arrays for a given action index"""
        set_arrays = []
        for k in range(1, len(action_indices)):
            sweep_indices = action_indices[:k]
    
            if sweep_indices in self.set_arrays:
                set_arrays.append(self.set_arrays[sweep_indices])
                # TODO handle grouped arrays (e.g. ParameterNode, nested Measurement)
        # Create new set array(s) if parameter result is an array or list
        if isinstance(result, (np.ndarray, list)):
            if isinstance(result, list):
                result = np.ndarray(result)
    
            for k, shape in enumerate(result.shape):
                arr = np.arange(shape)
                label = None
                unit = None
                if parameter is not None and hasattr(parameter, 'setpoints') \
                        and parameter.setpoints is not None:
                    arr_idx = parameter.names.index(name)
                    arr = parameter.setpoints[arr_idx][k]
                    label = parameter.setpoint_labels[arr_idx][k]
                    unit = parameter.setpoint_units[arr_idx][k]
    
                # Add singleton dimensions
                arr = np.broadcast_to(arr, result.shape[: k + 1])

                set_array = self._create_data_array(
                    action_indices=action_indices + (0,) * k,
                    result=arr,
                    name=f"{name}_set{k}",
                    label=label,
                    unit=unit,
                    is_setpoint=True,
                )
                set_arrays.append(set_array)

        # Add a dummy array in case the measurement was performed outside of
        # a Sweep. This is not needed if the result is an array
        if not set_arrays and not self.loop_indices:
            set_arrays = [
                self._create_data_array(
                    action_indices=running_measurement().action_indices,
                    result=result,
                    name="None",
                    is_setpoint=True,
                )
            ]
            set_arrays[0][0] = 1

        return tuple(set_arrays)

    def get_arrays(self, action_indices: Sequence[int] = None) -> List[DataArray]:
        """Get all arrays belonging to the current action indices

        If the action indices corresponds to a group of arrays (e.g. a nested
        measurement or ParameterNode), all the arrays in the group are returned

        Args:
            action_indices: Action indices of arrays.
                If not provided, the current action_indices are chosen

        Returns:
            List of data arrays matching the action indices
        """
        if action_indices is None:
            action_indices = self.action_indices

        if not isinstance(action_indices, Sequence):
            raise SyntaxError("parent_action_indices must be a tuple")

        num_indices = len(action_indices)
        return [
            arr
            for action_indices, arr in self.data_arrays.items()
            if action_indices[:num_indices] == action_indices
        ]


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
                self.data_handler = DataHandler()

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

                self.log(f'Measurement started {self.dataset.location}')
                print(f'Measurement started {self.dataset.location}')

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
            self.data_handler.add_metadata({"t_stop": t_stop})
            self.data_handler.add_metadata({"timings": self.timings})
            self.data_handler.finalize()

            self.log(f'Measurement finished')

        else:
            msmt.step_out(reduce_dimension=False)

        self.is_context_manager = False

    # TODO Needs to be implemented
    def _initialize_metadata(self, dataset):
        """Initialize dataset metadata"""
        if dataset is None:
            dataset = self.dataset

        config = qcodes_config
        dataset.add_metadata({"config": config})

        dataset.add_metadata({"measurement_type": "Measurement"})

        # Add instrument information
        if Station.default is not None:
            dataset.add_metadata({"station": Station.default.snapshot()})

        if using_ipython():
            measurement_cell = get_last_input_cells(1)[0]

            measurement_code = measurement_cell
            # If the code is run from a measurement thread, there is some
            # initial code that should be stripped
            init_string = "get_ipython().run_cell_magic('new_job', '', "
            if measurement_code.startswith(init_string):
                measurement_code = measurement_code[len(init_string) + 1 : -4]

            self._t_start = datetime.now()
            dataset.add_metadata(
                {
                    "measurement_cell": measurement_cell,
                    "measurement_code": measurement_code,
                    "last_input_cells": get_last_input_cells(20),
                    "t_start": self._t_start.strftime('%Y-%m-%d %H:%M:%S')
                }
            )

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
        elif isinstance(value, np.bool_):
            value = bool(value)

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


# TODO Any mention of set array should be changed
class Sweep:
    """Sweep over an iterable inside a Measurement

    Args:
        sequence: Sequence to iterate over.
            Can be an iterable, or a parameter Sweep.
            If the sequence
        name: Name of sweep. Not needed if a Parameter is passed
        unit: unit of sweep. Not needed if a Parameter is passed
        reverse: Sweep over sequence in opposite order.
            The data is also stored in reverse.
        restore: Stores the state of a parameter before sweeping it,
            then restores the original value upon exiting the loop.

    Examples:
        ```
        with Measurement('sweep_msmt') as msmt:
            for value in Sweep(np.linspace(5), 'sweep_values'):
                msmt.measure(value, 'linearly_increasing_value')

            p = Parameter('my_parameter')
            for param_val in Sweep(p.
        ```
    """
    def __init__(self, sequence, name=None, unit=None, reverse=False, restore=False):
        if running_measurement() is None:
            raise RuntimeError("Cannot create a sweep outside a Measurement")

        if not isinstance(sequence, Iterable):
            raise SyntaxError("Sweep sequence must be iterable")

        # Properties for the data array
        self.name = name
        self.unit = unit

        self.sequence = sequence
        self.dimension = len(running_measurement().loop_shape)
        self.loop_index = None
        self.iterator = None
        self.reverse = reverse
        self.restore = restore

        msmt = running_measurement()
        if msmt.action_indices in msmt.set_arrays:
            self.set_array = msmt.set_arrays[msmt.action_indices]
        else:
            self.set_array = self.create_set_array()

    def __iter__(self):
        if threading.current_thread() is not MeasurementLoop.measurement_thread:
            raise RuntimeError(
                "Cannot create a Sweep while another measurement "
                "is already running in a different thread."
            )
        if self.restore:
            if isinstance(self.sequence, SweepValues):
                running_measurement().mask(self.sequence.parameter, self.sequence.parameter.get())
            else:
                raise NotImplementedError("Unable to restore non-parameter values.")
        if self.reverse:
            self.loop_index = len(self.sequence) - 1
            self.iterator = iter(self.sequence[::-1])
        else:
            self.loop_index = 0
            self.iterator = iter(self.sequence)

        running_measurement().loop_shape += (len(self.sequence),)
        running_measurement().loop_indices += (self.loop_index,)
        running_measurement().action_indices += (0,)


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
            if self.restore:
                if isinstance(self.sequence, SweepValues):
                    msmt.unmask(self.sequence.parameter)
                else:
                    # TODO: Check what other iterators might be able to be masked
                    pass
            self.exit_sweep()

        if isinstance(self.sequence, SweepValues):
            self.sequence.set(sweep_value)

        self.set_array[msmt.loop_indices] = sweep_value

        self.loop_index += 1 if not self.reverse else -1

        return sweep_value

    def exit_sweep(self):
        msmt = running_measurement()
        msmt.step_out(reduce_dimension=True)
        raise StopIteration

    def create_set_array(self):
        if isinstance(self.sequence, SweepValues):
            return running_measurement()._create_data_array(
                action_indices=running_measurement().action_indices,
                result=self.sequence,
                parameter=self.sequence.parameter,
                is_setpoint=True,
            )
        else:
            return running_measurement()._create_data_array(
                action_indices=running_measurement().action_indices,
                result=self.sequence,
                name=self.name or "iterator",
                unit=self.unit,
                is_setpoint=True,
            )