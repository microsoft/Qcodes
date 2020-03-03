import sys
import numpy as np
from typing import List, Tuple, Union, Sequence, Dict, Any, Callable
import threading
from time import sleep

from qcodes.data.data_set import new_data, DataSet
from qcodes.data.data_array import DataArray
from qcodes.instrument.sweep_values import SweepValues
from qcodes.instrument.parameter import Parameter, MultiParameter
from qcodes.instrument.parameter_node import ParameterNode
from qcodes.utils.helpers import (
    using_ipython,
    directly_executed_from_cell,
    get_last_input_cells
)


class Measurement:
    """
    Args:
        name: Measurement name, also used as the dataset name
        force_cell_thread: Enforce that the measurement has been started from a
            separate thread if it has been directly executed from an IPython
            cell/prompt. This is because a measurement is usually run from a
            separate thread using the magic command `%%new_job`.
            An error is raised if this has not been satisfied.
            Note that if the measurement is started within a function, no error
            is raised.


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

    def __init__(self, name: str, force_cell_thread: bool = True):
        self.name = name

        # Total dimensionality of loop
        self.loop_shape: Union[Tuple[int], None] = None

        # Current loop indices
        self.loop_indices: Union[Tuple[int], None] = None

        # Index of current action
        self.action_indices: Union[Tuple[int], None] = None

        # contains data groups, such as ParameterNodes and nested measurements
        self._data_groups: Dict[Tuple[int], "Measurement"] = {}

        # Registry of actions: sweeps, measurements, and data groups
        self.actions: Dict[Tuple[int], Any] = {}

        self.is_context_manager: bool = False  # Whether used as context manager
        self.is_paused: bool = False  # Whether the Measurement is paused
        self.is_stopped: bool = False  # Whether the Measurement is stopped

        self.force_cell_thread = force_cell_thread and using_ipython()

    @property
    def data_groups(self) -> Dict[Tuple[int], "Measurement"]:
        if running_measurement() is not None:
            return running_measurement()._data_groups
        else:
            return self._data_groups

    @property
    def active_action(self):
        return self.actions.get(self.action_indices, None)

    def __enter__(self):
        self.is_context_manager = True

        # Encapsulate everything in a try/except to ensure that the context
        # manager is properly exited.
        try:
            if Measurement.running_measurement is None:
                # Register current measurement as active primary measurement
                Measurement.running_measurement = self
                Measurement.measurement_thread = threading.current_thread()

                # Initialize dataset
                self.dataset = new_data(name=self.name)
                self.dataset.add_metadata({"measurement_type": "Measurement"})
                self.dataset.active = True

                self._initialize_metadata(self.dataset)

                # Initialize attributes
                self.loop_shape = ()
                self.loop_indices = ()
                self.action_indices = (0,)
                self.data_arrays = {}
                self.set_arrays = {}

            else:
                if threading.current_thread() is not Measurement.measurement_thread:
                    raise RuntimeError(
                        "Cannot run a measurement while another measurement "
                        "is already running in a different thread."
                    )

                # Primary measurement is already running. Add this measurement as
                # a data_group of the primary measurement
                msmt = Measurement.running_measurement
                msmt.data_groups[msmt.action_indices] = self
                msmt.action_indices += (0,)

                # Nested measurement attributes should mimic the primary measurement
                self.loop_shape = msmt.loop_shape
                self.loop_indices = msmt.loop_indices
                self.action_indices = msmt.action_indices
                self.data_arrays = msmt.data_arrays
                self.set_arrays = msmt.set_arrays

            # Perform measurement thread check, and set user namespace variables
            if self.force_cell_thread and Measurement.running_measurement is self:
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
                shell.user_ns[self._default_dataset_name] = self.dataset

            return self
        except:
            # An error has occured, ensure running_measurement is cleared
            if Measurement.running_measurement is self:
                Measurement.running_measurement = None
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        msmt = Measurement.running_measurement
        if msmt is self:
            Measurement.running_measurement = None
            self.dataset.finalize()
            self.dataset.active = False
        else:
            # This is a nested measurement.
            # update action_indices of primary measurements
            msmt.action_indices = msmt.action_indices[:-1]

        self.is_context_manager = False

    def _initialize_metadata(self, dataset: DataSet = None):
        if dataset is None:
            dataset = self.dataset

        if using_ipython():
            measurement_cell = get_last_input_cells(1)[0]

            measurement_code = measurement_cell
            # If the code is run from a measurement thread, there is some
            # initial code that should be stripped
            init_string = "get_ipython().run_cell_magic('new_job', '', "
            if measurement_code.startswith(init_string):
                measurement_code = measurement_code[len(init_string)+1:-4]

            dataset.add_metadata({
                'measurement_cell': measurement_cell,
                'measurement_code': measurement_code,
                'last_input_cells': get_last_input_cells(20)
            })

    # Data array functions
    def _create_data_array(
        self,
        action_indices: Tuple[int],
        result,
        parameter: Parameter = None,
        ndim: int = None,
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
            ndim: Number of dimensions. If not provided, will use length of
                action_indices
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

        if ndim is None:
            ndim = len(action_indices)

        array_kwargs = {
            "is_setpoint": is_setpoint,
            "action_indices": action_indices,
            "shape": self.loop_shape,
        }

        if is_setpoint or isinstance(result, (np.ndarray, list)):
            array_kwargs["shape"] += np.shape(result)

        if isinstance(parameter, Parameter):
            array_kwargs["parameter"] = parameter
            # Add a custom name
            if name is not None:
                array_kwargs["full_name"] = name
        else:
            array_kwargs["name"] = name
            if label is None:
                label = name[0].capitalize() + name[1:].replace("_", " ")
            array_kwargs["label"] = label
            array_kwargs["unit"] = unit or ""

        # Add setpoint arrays
        if not is_setpoint:
            array_kwargs["set_arrays"] = self._add_set_arrays(
                action_indices, result, name=(name or parameter.name), ndim=ndim
            )

        data_array = DataArray(**array_kwargs)

        data_array.array_id = data_array.full_name
        data_array.array_id += "_" + "_".join(str(k) for k in action_indices)

        data_array.init_data()

        self.dataset.add_array(data_array)

        # Add array to set_arrays or to data_arrays of this Measurement
        if is_setpoint:
            self.set_arrays[action_indices] = data_array
        else:
            self.data_arrays[action_indices] = data_array

        return data_array

    def _add_set_arrays(
        self, action_indices: Tuple[int], result, name: str, ndim: int,
    ):
        set_arrays = []
        for k in range(1, ndim):
            sweep_indices = action_indices[:k]
            if sweep_indices in self.set_arrays:
                set_arrays.append(self.set_arrays[sweep_indices])
                # TODO handle grouped arrays (e.g. ParameterNode, nested Measurement)

        # Create new set array(s) if parameter result is an array or list
        if isinstance(result, (np.ndarray, list)):
            if isinstance(result, list):
                result = np.ndarray(result)

            # TODO handle if the parameter contains attribute setpoints

            for k, shape in enumerate(result.shape):
                arr = np.arange(shape)
                # Add singleton dimensions
                arr = np.broadcast_to(arr, result.shape[: k + 1])

                set_array = self._create_data_array(
                    action_indices=action_indices + (0,) * k,
                    result=arr,
                    name=f"{name}_set{k}",
                    is_setpoint=True,
                )
                set_arrays.append(set_array)

        return tuple(set_arrays)

    # def _add_data_group(self, data_group):

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

    def _verify_action(self, action, name=None, add_if_new=True):
        """Verify an action corresponds to the current action indices.

        This is only relevant if an action has previously been performed at
        these action indices
        """
        if self.action_indices not in self.actions and add_if_new:
            # Add current action to action registry
            self.actions[self.action_indices] = action
        else:
            existing_action = self.actions[self.action_indices]
            if isinstance(existing_action, str):
                # Action is a measurement of a raw value, not via a parameter.
                name_action = existing_action
                name = name or action
            else:
                name = name or action.name
                if hasattr(existing_action, 'name'):
                    name_action = existing_action.name
                elif hasattr(existing_action, '__name__'):
                    name_action = existing_action.__name__
                else:
                    raise RuntimeError(f'Existing action {existing_action} has no name')

            if name_action != name:
                raise RuntimeError(
                    f'Wrong measurement at action_indices {self.action_indices}. '
                    f'Expected: {name_action}. Received: {name}'
                )

    def _add_measurement_result(
        self,
        action_indices,
        result,
        parameter=None,
        ndim=None,
        store: bool = True,
        name: str = None,
        label: str = None,
        unit: str = None,
    ):
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
                ndim=ndim,
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
                # Add singleton dimensions
                arr = np.broadcast_to(arr, result.shape[: k + 1])
                data_to_store[set_array.array_id] = arr

        if store:
            self.dataset.store(self.loop_indices, data_to_store)

        return data_to_store

    # Measurement-related functions
    def _measure_parameter(self, parameter, name=None):
        # Ensure measuring parameter matches the current action_indices
        self._verify_action(action=parameter, name=name, add_if_new=True)

        # Get parameter result
        result = parameter()

        self._add_measurement_result(
            self.action_indices, result, parameter=parameter, name=name
        )

        return result

    def _measure_multi_parameter(self, multi_parameter, name=None):
        # Ensure measuring multi_parameter matches the current action_indices
        self._verify_action(action=multi_parameter, name=name, add_if_new=True)

        results_list = multi_parameter()

        results = {
            name: result for name, result in zip(multi_parameter.names, results_list)
        }

        if name is None:
            name = multi_parameter.name

        # TODO also incorporate setpoints
        with Measurement(name) as msmt:
            for k, (key, val) in enumerate(results.items()):
                msmt.measure(
                    val,
                    name=key,
                    label=multi_parameter.labels[k],
                    unit=multi_parameter.units[k],
                )

        return results

    def _measure_callable(self, callable, name=None):
        # Determine name
        if name is None:
            if hasattr(callable, "__self__") and isinstance(
                callable.__self__, ParameterNode
            ):
                name = callable.__self__.name
            elif hasattr(callable, "__name__"):
                name = callable.__name__
            else:
                action_indices_str = "_".join(
                    str(idx) for idx in self.action_indices
                )
                name = f"data_group_{action_indices_str}"

        # Ensure measuring callable matches the current action_indices
        self._verify_action(action=callable, name=name, add_if_new=True)

        results = callable()

        # Check if the callable already performed a nested measurement
        # In this case, the nested measurement is stored as a data_group, and
        # has loop indices corresponding to the current ones.
        msmt = Measurement.running_measurement
        data_group = msmt.data_groups.get(self.action_indices)
        if getattr(data_group, "loop_indices", None) == self.loop_indices:
            # Measurement has already been performed by a nested measurement
            return results
        else:
            # No nested measurement has been performed in the callable.
            # Add results, which should be dict, by creating a nested measurement
            if not isinstance(results, dict):
                raise SyntaxError(f"{name} results must be a dict, not {results}")

            with Measurement(name) as msmt:
                for key, val in results.items():
                    msmt.measure(val, name=key)

        return results

    def _measure_value(self, value, name):
        # Ensure measuring callable matches the current action_indices
        self._verify_action(action=name, add_if_new=True)

        result = value
        self._add_measurement_result(
            action_indices=self.action_indices,
            result=result,
            name=name,
            # label=label,
            # unit=unit,
        )
        # TODO uncomment label, unit

    def measure(
        self,
        measurable: Union[Parameter, Callable, float, int, bool, np.ndarray],
        name=None,
        label=None,
        unit=None,
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

        Returns:
            Return value of measurable
        """
        # TODO add label, unit, etc. as kwargs
        if not self.is_context_manager:
            raise RuntimeError(
                "Must use the Measurement as a context manager, "
                "i.e. 'with Measurement(name) as msmt:'"
            )
        elif self.is_stopped:
            raise SystemExit("Measurement.stop() has been called")
        elif threading.current_thread() is not Measurement.measurement_thread:
            raise RuntimeError(
                "Cannot measure while another measurement is already running "
                "in a different thread."
            )

        if self != Measurement.running_measurement:
            # Since this Measurement is not the running measurement, it is a
            # DataGroup in the running measurement. Delegate measurement to the
            # running measurement
            return Measurement.running_measurement.measure(measurable, name=name)

        # Code from hereon is only reached by the primary measurement,
        # i.e. the running_measurement

        # Wait as long as the measurement is paused
        while self.is_paused:
            sleep(0.1)

        # TODO Incorporate kwargs name, label, and unit, into each of these
        if isinstance(measurable, Parameter):
            result = self._measure_parameter(measurable, name=name)
        elif isinstance(measurable, MultiParameter):
            result = self._measure_multi_parameter(measurable, name=name)
        elif callable(measurable):
            result = self._measure_callable(measurable, name=name)
        elif isinstance(measurable, (float, int, bool, np.ndarray)):
            result = self._measure_value(measurable, name=name)
        else:
            raise RuntimeError(
                f"Cannot measure {measurable} as it cannot be called, and it "
                f"is not an int, float, bool, or numpy array."
            )

        # Increment last action index by 1
        self.skip()

        return result

    # Functions relating to measurement flow
    def pause(self):
        """Pause measurement at start of next parameter sweep/measurement"""
        self.is_paused = True

    def resume(self):
        """Resume measurement after being paused"""
        self.is_paused = False

    def stop(self):
        self.is_stopped = True
        # Unpause loop
        self.resume()

    def skip(self, N=1):
        action_indices = list(self.action_indices)
        action_indices[-1] += N
        self.action_indices = tuple(action_indices)
        return self.action_indices


    def exit_loop(self):
        if Measurement.running_measurement is not self:
            Measurement.running_measurement.exit_loop()
        else:
            self.loop_shape = self.loop_shape[:-1]
            self.loop_indices = self.loop_indices[:-1]

            # Remove last action index and increment one before that by one
            action_indices = list(self.action_indices[:-1])
            action_indices[-1] += 1
            self.action_indices = tuple(action_indices)


def running_measurement() -> Measurement:
    return Measurement.running_measurement


class Sweep:
    def __init__(self, sequence, name=None, unit=None):
        if running_measurement() is None:
            raise RuntimeError("Cannot create a sweep outside a Measurement")

        # Properties for the data array
        self.name = name
        self.unit = unit

        self.sequence = sequence
        self.dimension = len(running_measurement().loop_shape)
        self.loop_index = None
        self.iterator = None

        if running_measurement().action_indices in running_measurement().set_arrays:
            self.set_array = running_measurement().set_arrays[
                running_measurement().action_indices
            ]
        else:
            self.set_array = self.create_set_array()

    def __iter__(self):
        if threading.current_thread() is not Measurement.measurement_thread:
            raise RuntimeError(
                "Cannot create a Sweep while another measurement "
                "is already running in a different thread."
            )

        running_measurement().loop_shape += (len(self.sequence),)
        running_measurement().loop_indices += (0,)
        running_measurement().action_indices += (0,)

        # Create a set array if necessary

        self.loop_index = 0
        self.iterator = iter(self.sequence)

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
            msmt.exit_loop()
            raise StopIteration

        if isinstance(self.sequence, SweepValues):
            self.sequence.set(sweep_value)

        self.set_array[msmt.loop_indices] = sweep_value

        self.loop_index += 1

        return sweep_value

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
