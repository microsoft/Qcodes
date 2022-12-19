import builtins
import json
import logging
import threading
import traceback
from datetime import datetime
from time import perf_counter, sleep
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from warnings import warn
from matplotlib import pyplot as plt

import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from qcodes.dataset.data_set_protocol import DataSetProtocol
from qcodes.dataset.descriptions.detect_shapes import detect_shape_of_measurement
from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.descriptions.versioning import serialization as serial
from qcodes.dataset.descriptions.versioning.converters import new_to_old
from qcodes.dataset.dond.sweeps import AbstractSweep
from qcodes.dataset.measurements import DataSaver, Measurement, Runner
from qcodes.dataset.sqlite.queries import add_parameter, update_run_description
from qcodes.instrument import (
    DelegateParameter,
    InstrumentBase,
    MultiParameter,
    Parameter,
    SweepValues,
)
from qcodes.instrument.parameter import _BaseParameter
from qcodes.parameters import ParameterBase
from qcodes.station import Station
from qcodes.utils import NumpyJSONEncoder
from qcodes.utils.dataset.doNd import AbstractSweep, ActionsT
from qcodes.utils.helpers import PerformanceTimer

RAW_VALUE_TYPES = (
    float,
    int,
    bool,
    np.integer,
    np.floating,
    np.bool_,
    type(None),
)


class _DatasetHandler:
    """Handler for a single DataSet (with Measurement and Runner)

    Used by the `MeasurementLoop` as an interface to the `Measurement` and `DataSet`
    """

    def __init__(self, measurement_loop: "MeasurementLoop", name: str = "results"):
        self.measurement_loop = measurement_loop
        self.name = name

        self.initialized: bool = False
        self.datasaver: Optional[DataSaver] = None
        self.runner: Optional[Runner] = None
        self.measurement: Optional[Measurement] = None
        self.dataset: Optional[DataSetProtocol] = None

        # Key: action_index
        # Values:
        # - parameter
        # - dataset_parameter (differs from "parameter" when multiple share same name)
        # - latest_value
        self.setpoint_list: Dict[Tuple[int], Any] = dict()

        # Dict with key being action_index and value is a dict containing
        # - parameter
        # - setpoints_action_indices
        # - setpoint_parameters
        # - shape
        # - unstored_results - list where each element contains (*setpoints, measurement_value)
        # - latest_value
        self.measurement_list: Dict[Tuple[int], Any] = {}

        self.initialize()

    def initialize(self) -> None:
        """Creates a `Measurement`, runs it and initializes a dataset"""
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

    def finalize(self) -> None:
        """Finishes a measurement by flushing all data to the database"""
        self.datasaver.flush_data_to_database()

    def _ensure_unique_parameter(
        self, parameter_info: dict, setpoint: bool, max_idx: int = 99
    ) -> None:
        """Ensure setpoint / measurement parameters have unique names

        If a previously registered parameter already shares the same name, it adds a
        suffix '{name}_{idx}' where idx starts at zero

        Args:
            parameter_info: dict for a setpoint/measurement parameter
                See `DatasetHandler.create_measurement_info` for more information
            setpoints: Whether parameter is a setpoint
            max_idx: maximum allowed incremental index when parameters share same name

        Raises:
            OverflowError if more than ``max_idx`` parameters share the same name
        """
        if setpoint:
            parameter_list = self.setpoint_list
        else:
            parameter_list = self.measurement_list

        parameter_names = [
            param_info["dataset_parameter"].name
            for param_info in parameter_list.values()
            if "dataset_parameter" in param_info
        ]

        parameter_name = parameter_info["parameter"].name
        if parameter_name not in parameter_names:
            parameter_info["dataset_parameter"] = parameter_info["parameter"]
        else:
            for idx in range(1, max_idx):
                parameter_idx_name = f"{parameter_name}_{idx}"
                if parameter_idx_name not in parameter_names:
                    parameter_name = parameter_idx_name
                    break
            else:
                raise OverflowError(
                    f"All parameter names {parameter_name}_{{idx}} up to idx {max_idx} are taken"
                )
            # Create a delegate parameter with modified name
            delegate_parameter = DelegateParameter(
                name=parameter_name, source=parameter_info["parameter"]
            )
            parameter_info["dataset_parameter"] = delegate_parameter

    def create_measurement_info(
        self,
        action_indices: Tuple[int],
        parameter: Parameter,
        name: Optional[str] = None,
        label: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates information dict for a parameter that is to be measured

        Args:
            action_indices: Indices in measurement loop corresponding to the
                parameter being measured.
            parameter: Parameter to be measured.
            name: Name used for the measured parameter.
                Will use parameter.name if not provided.
            label: Label used for the measured parameter.
                Will use parameter.label if not provided.
            unit: Unit used for the measured parameter.
                Will use parameter.unit if not provided.
        """
        if parameter is None:
            assert name is not None
            parameter = Parameter(name=name, label=label, unit=unit)
        elif {name, label, unit} != {
            None,
        }:
            overwrite_attrs = {"name": name, "label": label, "unit": unit}
            overwrite_attrs = {
                key: val for key, val in overwrite_attrs.items() if val is not None
            }
            parameter = DelegateParameter(source=parameter, **overwrite_attrs)

        setpoints_action_indices = []
        for k in range(len(action_indices) + 1):
            if action_indices[:k] in self.setpoint_list:
                setpoints_action_indices.append(action_indices[:k])

        measurement_info = {
            "parameter": parameter,
            "setpoints_action_indices": setpoints_action_indices,
            "shape": self.measurement_loop.loop_shape,
            "unstored_results": [],
            "registered": False,
        }

        return measurement_info

    def register_new_measurement(
        self,
        action_indices: Tuple[int],
        parameter: _BaseParameter,
        name: Optional[str] = None,
        label: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> None:
        """Register a new measurement parameter"""
        measurement_info = self.create_measurement_info(
            action_indices=action_indices,
            parameter=parameter,
            name=name,
            label=label,
            unit=unit,
        )
        self.measurement_list[action_indices] = measurement_info

        # Add new measurement parameter
        self._update_interdependencies()

    def add_measurement_result(
        self,
        action_indices: Tuple[int],
        result: Union[float, int, bool],
        parameter: _BaseParameter = None,
        name: Optional[str] = None,
        label: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> None:
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
                unit=unit,
            )

        measurement_info = self.measurement_list[action_indices]

        if name is None and parameter is not None:
            name = parameter.name
        if name != measurement_info["parameter"].name:
            raise SyntaxError(
                f"Provided name {name} must match that of previous measurement "
                f"{measurement_info['parameter'].name}"
            )

        # Get setpoints corresponding to measurement
        setpoints = self.get_result_setpoints(result, action_indices=action_indices)

        # Store results
        parameters = (
            *measurement_info["setpoint_parameters"],
            measurement_info["dataset_parameter"],
        )
        result_with_setpoints = tuple(zip(parameters, (*setpoints, result)))
        self.datasaver.add_result(*result_with_setpoints)

        # Also store in measurement_info
        measurement_info["latest_value"] = result

    def get_result_setpoints(self, result, action_indices):
        measurement_info = self.measurement_list[action_indices]
        # Check if result is an array
        if np.ndim(result) > 0:
            if len(measurement_info["setpoints_action_indices"]) < np.ndim(result):
                raise ValueError(
                    f"Number of setpoints {len(measurement_info['setpoints_action_indices'])} "
                    f"is less than array dimensionality {np.ndim(result)}"
                )

            # Pick the last N sweeps, where N is the array dimensionality
            setpoints_action_indices = measurement_info["setpoints_action_indices"]
            repeat_setpoints_action_indices = setpoints_action_indices[:-np.ndim(result)]
            mesh_setpoints_action_indices = setpoints_action_indices[-np.ndim(result):]

            # Create repetitions of outer setpoints
            repeat_setpoint_arrs = []
            for k, setpoint_indices in enumerate(repeat_setpoints_action_indices):
                latest_value = self.setpoint_list[setpoint_indices]["latest_value"]
                setpoint_arr = np.tile(latest_value, np.shape(result))
                repeat_setpoint_arrs.append(setpoint_arr)

            # Create mesh from last N setpoints matching 
            mesh_setpoint_arrs = []
            for k, setpoint_indices in enumerate(mesh_setpoints_action_indices):
                setpoint_info = self.setpoint_list[setpoint_indices]
                sequence = setpoint_info["sweep"].sequence
                mesh_setpoint_arrs.append(sequence)
                if len(sequence) != np.shape(result)[k]:
                    raise ValueError(
                        f'Setpoint {k} {setpoint_info["sweep"].name} length differs '
                        f'from dimension {k} of array: {len(sequence)=} != {np.shape(result)[k]=}'
                    )

            # Convert all 1D setpoint arrays to an N-D meshgrid
            setpoints = repeat_setpoint_arrs + list(np.meshgrid(*mesh_setpoint_arrs, indexing='ij'))
        else:
            setpoints = [
                self.setpoint_list[action_indices]["latest_value"]
                for action_indices in measurement_info["setpoints_action_indices"]
            ]
        
        return setpoints

    def _update_interdependencies(self) -> None:
        """Updates dataset after instantiation to include new setpoint/measurement parameter

        The `DataSet` was not made to register parameters after instantiation, so this
        method is non-intuitive.
        """
        dataset = self.datasaver.dataset

        # Get previous paramspecs
        previous_paramspecs = dataset._rundescriber.interdeps.paramspecs
        previous_paramspec_names = [spec.name for spec in previous_paramspecs]

        # Register all new setpoints parameters in Measurement
        for setpoint_info in self.setpoint_list.values():
            if setpoint_info["registered"]:
                # Already registered
                continue

            self._ensure_unique_parameter(setpoint_info, setpoint=True)
            self.measurement.register_parameter(setpoint_info["dataset_parameter"])
            setpoint_info["registered"] = True

        # Register all measurement parameters in Measurement
        for measurement_info in self.measurement_list.values():
            if measurement_info["registered"]:
                # Already registered
                continue

            # Determine setpoint_parameters for each measurement_parameter
            for measurement_info in self.measurement_list.values():
                measurement_info["setpoint_parameters"] = tuple(
                    self.setpoint_list[action_indices]["dataset_parameter"]
                    for action_indices in measurement_info["setpoints_action_indices"]
                )

            self._ensure_unique_parameter(measurement_info, setpoint=False)
            self.measurement.register_parameter(
                measurement_info["dataset_parameter"],
                setpoints=measurement_info["setpoint_parameters"],
            )
            measurement_info["registered"] = True
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
    """Class to perform measurements  in a fixed sequential order.

    This measurement method complements the other two ways of doing measurements
    by being more versatile than `do1d`, `do2d`, `dond`, and more implicit that `Measurement`.

    See the tutorial ``MeasurementLoop`` for a tutorial.

    Args:
        name: Measurement name, also used as the dataset name
        notify: Notify when measurement is complete.
            The function `Measurement.notify_function` must be set
        show_progress: Whether to show progress bars.
            If not specified, will use value of class attribute ``MeasurementLoop.show_progress``
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

    # Progress bar
    show_progress: bool = False
    _progress_bar_kwargs: Dict[str, Any] = {'mininterval': 0.2}

    _t_start = None

    # Notification function, called if notify=True.
    # Function should receive the following arguments:
    # Measurement object, exception_type, exception_message, traceback
    # The last three are only not None if an error has occured
    notify_function = None

    def __init__(self, name: Optional[str], notify: bool = False, show_progress: bool = None):
        self.name: str = name

        # Data handler is created during `with Measurement("name")`
        # Used to control dataset(s)
        self.data_handler: _DatasetHandler = None

        # Total dimensionality of loop
        self.loop_shape: Union[Tuple[int], None] = None

        # Current loop indices
        self.loop_indices: Union[Tuple[int], None] = None

        # Index of current action
        self.action_indices: Union[Tuple[int], None] = None

        # Progress bars, only used if show_progress is True
        if show_progress is not None:
            self.show_progress = show_progress
        self.progress_bars: Dict[Tuple[int], tqdm] = {}


        # contains data groups, such as ParameterNodes and nested measurements
        self._data_groups: Dict[Tuple[int], "MeasurementLoop"] = {}

        # Registry of actions: sweeps, measurements, and data groups
        self.actions: Dict[Tuple[int], Any] = {}
        self.action_names: Dict[Tuple[int], str] = {}

        self.is_context_manager: bool = False  # Whether used as context manager
        self.is_paused: bool = False  # Whether the Measurement is paused
        self.is_stopped: bool = False  # Whether the Measurement is stopped

        # Whether to notify upon measurement completion
        self.notify: bool = notify

        # Each measurement can have its own final actions, to be executed
        # regardless of whether the measurement finished successfully or not
        # Note that there are also Measurement.final_actions, which are always
        # executed when the outermost measurement finishes
        self.final_actions: List[Callable] = []
        self.except_actions: List[Callable] = []
        self._masked_properties: List[Dict[str, Any]] = []

        self.timings: PerformanceTimer = PerformanceTimer()

    @property
    def dataset(self) -> DataSetProtocol:
        if self.data_handler is None:
            return None
        else:
            return self.data_handler.dataset

    def log(self, message: str, level: str = "info") -> None:
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
    def active_action(self) -> Optional[Tuple[int]]:
        return self.actions.get(self.action_indices, None)

    @property
    def active_action_name(self) -> Optional[str]:
        return self.action_names.get(self.action_indices, None)

    @property
    def setpoint_list(self) -> Optional[Dict[Tuple[int], Any]]:
        if self.data_handler is not None:
            return self.data_handler.setpoint_list
        else:
            return None

    @property
    def measurement_list(self) -> Optional[Dict[Tuple[int], Any]]:
        if self.data_handler is not None:
            return self.data_handler.measurement_list
        else:
            return None

    def __enter__(self) -> "MeasurementLoop":
        """Operation when entering a loop, including dataset instantiation"""
        self.is_context_manager = True

        # Encapsulate everything in a try/except to ensure that the context
        # manager is properly exited.
        try:
            if MeasurementLoop.running_measurement is None:
                # Register current measurement as active primary measurement
                MeasurementLoop.running_measurement = self
                MeasurementLoop.measurement_thread = threading.current_thread()

                # Initialize dataset handler
                self.data_handler = _DatasetHandler(
                    measurement_loop=self, name=self.name
                )

                # Add metadata
                self._t_start = datetime.now()
                self._initialize_metadata(self.dataset)

                # Initialize attributes
                self.loop_shape = ()
                self.loop_indices = ()
                self.action_indices = (0,)
                self.data_arrays = {}
                self.set_arrays = {}

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
                # data_groups = [
                #     (key, getattr(val, "name", "None"))
                #     for key, val in msmt.data_groups.items()
                # ]
                # TODO add metadata
                # msmt.dataset.add_metadata({"data_groups": data_groups})
                msmt.action_indices += (0,)

                # Nested measurement attributes should mimic the primary measurement
                self.loop_shape = msmt.loop_shape
                self.loop_indices = msmt.loop_indices
                self.action_indices = msmt.action_indices
                self.data_arrays = msmt.data_arrays
                self.set_arrays = msmt.set_arrays
                self.timings = msmt.timings

            return self
        except:
            # An error has occured, ensure running_measurement is cleared
            if MeasurementLoop.running_measurement is self:
                MeasurementLoop.running_measurement = None
            raise

    def __exit__(self, exc_type: Exception, exc_val, exc_tb) -> None:
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

            for progress_bar in self.progress_bars.values():
                progress_bar.close()

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
                except Exception:
                    self.log("Could not notify", level="error")

            # include final metadata
            t_stop = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.dataset.add_metadata("t_stop", t_stop)
            self.data_handler.finalize()

            self.log("Measurement finished")
        else:
            msmt.step_out(reduce_dimension=False)

        self.is_context_manager = False

    def _initialize_metadata(self, dataset):
        """Initialize dataset metadata"""
        if dataset is None:
            dataset = self.dataset

        # Save config to metadata
        try:
            from qcodes import config

            config_str = json.dumps(dict(config), cls=NumpyJSONEncoder)
            self.dataset.add_metadata('config', config_str)
        except Exception as e:
            warn(f'Could not save config due to error {e}')

        dataset.add_metadata("measurement_type", "MeasurementLoop")
        dataset.add_metadata("t_start", self._t_start.strftime("%Y-%m-%d %H:%M:%S"))

        # Save latest IPython cells
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None and "In" in shell.ns_table["user_global"]:
            num_cells = 20  # Number of cells to save
            last_input_cells = shell.ns_table["user_global"]['In'][-num_cells:]
            dataset.add_metadata("measurement_code", last_input_cells[-1])
            dataset.add_metadata("last_input_cells", str(last_input_cells))

    def _verify_action(
        self, action: Callable, name: str, add_if_new: bool = True
    ) -> None:
        """Verify an action corresponds to the current action indices.

        An action is usually (currently always) a measurement.

        Args:
            action: Action that is supposed to be performed at these action_indices
            add_if_new: Register action if the action_indices have not yet been registered

        Raises:
            RuntimeError if a different action is performed than is usually
                performed at the current action_indices. An example is when
                a different parameter is measuremed.
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

    def _apply_actions(self, actions: list, label="", clear=False) -> None:
        """Apply actions, either except_actions or final_actions"""
        for action in actions:
            try:
                action()
            except Exception:
                self.log(
                    f"Could not execute {label} action {action} \n"
                    f"{traceback.format_exc()}",
                    level="error",
                )

        if clear:
            actions.clear()

    def _get_maximum_action_index(self, action_indices, position):
        msmt = running_measurement()

        # Get maximum action idx
        max_idx = 0
        for idxs in msmt.actions:
            if idxs[:position] != action_indices[:position]:
                continue
            if len(idxs) <= position:
                continue
            max_idx = max(max_idx, idxs[position])
        return max_idx

    def _update_progress_bar(self, action_indices, description=None, create_if_new=True):
        # Register new progress bar
        if action_indices not in self.progress_bars:
            # Do not create progress bar if one already exists and it's not a widget
            # Otherwise stdout gets spammed
            if not isinstance(tqdm, tqdm_notebook) and self.progress_bars:
                return
            elif create_if_new:
                self.progress_bars[action_indices] = tqdm(
                    total=np.prod(self.loop_shape),
                    desc=description,
                    **self._progress_bar_kwargs
                )
            else:
                raise RuntimeError('Cannot update progress bar if not created')

        # Update progress bar
        progress_bar = self.progress_bars[action_indices]
        value = 1
        for k, loop_idx in enumerate(self.loop_indices[::-1]):
            if k:
                factor = np.prod(self.loop_shape[-k:])
            else:
                factor = 1
            value += factor * loop_idx

        progress_bar.update(value - progress_bar.n)
        if value == progress_bar.total:
            progress_bar.close()


    def _fraction_complete_action_indices(self, action_indices, silent=True):
        """Calculate fraction complete from finished action_indices"""
        msmt = running_measurement()
        fraction_complete = 0
        scale = 1

        max_idxs = []
        for k, action_idx in enumerate(action_indices):
            # Check if previous idx is a sweep
            # If so, reduce scale by loop dimension
            action = msmt.actions.get(action_indices[:k])
            if not silent:
                print(f'{action=}, {isinstance(action, BaseSweep)=}')
            if isinstance(action, BaseSweep):
                if not silent:
                    print(f'Decreasing scale by {len(action)}')
                scale /= len(action)

            max_idx = self._get_maximum_action_index(action_indices, position=k)

            fraction_complete += action_idx / (max_idx + 1) * scale
            scale /= max_idx + 1
            max_idxs.append(max_idx)
            if not silent:
                print(f'{fraction_complete=}, {scale=}, {action_idx=}, {max_idxs=}')

        return fraction_complete

    def _fraction_complete_loop(self, action_indices, silent=True):
        msmt = running_measurement()
        fraction_complete = 0
        scale = 1
        loop_idx = 0

        for k, action_idx in enumerate(action_indices):
            # Check if current action is a sweep
            # If so, reduce scale by action index fraction
            action = msmt.actions.get(action_indices[:k+1])
            if isinstance(action, BaseSweep):
                max_idx = self._get_maximum_action_index(action_indices, position=k)

                if not silent:
                    print(f'Reducing current Sweep {loop_idx=} {msmt.loop_indices[loop_idx]} / {len(action)} * {scale}')
                    print(f'{max_idx=}')
                scale /= (max_idx + 1)

            # Check if previous idx is a sweep
            # If so, reduce scale by loop dimension
            action = msmt.actions.get(action_indices[:k])
            if not silent:
                print(f'{action=}, {isinstance(action, BaseSweep)=}')
            if isinstance(action, BaseSweep):
                if not silent:
                    print(f'Reducing previous Sweep {loop_idx=} fraction {msmt.loop_indices[loop_idx]} / {len(action)} * {scale}')
                fraction_complete += msmt.loop_indices[loop_idx] / len(action) * scale
                loop_idx += 1
                scale /= len(action)

        return fraction_complete

    def fraction_complete(self, silent=True, precision=3):
        msmt = running_measurement()
        if msmt is None:
            return 1

        fraction_complete = 0

        # Calculate fraction complete from action indices
        fraction_complete_actions = self._fraction_complete_action_indices(msmt.action_indices, silent=silent+1)
        fraction_complete += fraction_complete_actions
        if not silent:
            print(f'Fraction complete from action indices: {fraction_complete_actions:.3f}')

        # Calculate fraction complete from point in loop
        fraction_complete_loop = self._fraction_complete_loop(msmt.action_indices, silent=silent+1)
        fraction_complete += fraction_complete_loop
        if not silent:
            print(f'Fraction complete from loop: {fraction_complete_loop:.3f}')

        return np.round(fraction_complete, precision)

    # Measurement-related functions
    def _measure_parameter(
        self,
        parameter: _BaseParameter,
        name: Optional[str] = None,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Measure parameter and store results.

        Called from `measure`.
        MultiParameter is called separately.

        Args:
            parameter: Parameter to be measured
            name: Name used to measure parameter, overriding ``parameter.name``
            label: Label used to measure parameter, overriding ``parameter.label``
            unit: Unit used to measure parameter, overriding ``parameter.unit``
            **kwargs: optional kwargs passed to parameter, i.e. ``parameter(**kwargs)``

        Returns:
            Current value of parameter
        """
        name = name or parameter.name

        # Ensure measuring parameter matches the current action_indices
        self._verify_action(action=parameter, name=name, add_if_new=True)

        # Get parameter result
        result = parameter(**kwargs)

        # Result "None causes issues, so it's converted to NaN"
        if result is None:
            result = np.nan

        self.data_handler.add_measurement_result(
            action_indices=self.action_indices,
            result=result,
            parameter=parameter,
            name=name,
            label=label,
            unit=unit,
        )

        return result

    def _measure_multi_parameter(
        self, multi_parameter: MultiParameter, name: str = None, **kwargs
    ) -> Any:
        """Measure MultiParameter and store results

        Called from `measure`

        Args:
            parameter: Parameter to be measured
            name: Name used to measure parameter, overriding ``parameter.name``
            **kwargs: optional kwargs passed to parameter, i.e. ``parameter(**kwargs)``

        Returns:
            Current value of parameter

        Notes:
            - Does not store setpoints yet
        """
        name = name or multi_parameter.name

        # Ensure measuring multi_parameter matches the current action_indices
        self._verify_action(action=multi_parameter, name=name, add_if_new=True)

        with self.timings.record(["measurement", self.action_indices, "get"]):
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

    def _measure_callable(
        self, measurable_function: Callable, name: str = None, **kwargs
    ) -> Dict[str, Any]:
        """Measure a callable (function) and store results

        The function should return a dict, from which each item is measured.
        If the function already contains creates a Measurement, the return
        values aren't stored.

        Args:
            name: Dataset name used for function. Extracts name from function if not provided
            **kwargs: optional kwargs passed to callable, i.e. ``callable(**kwargs)``
        """
        # Determine name
        if name is None:
            if hasattr(measurable_function, "__self__") and isinstance(
                measurable_function.__self__, InstrumentBase
            ):
                name = measurable_function.__self__.name
            elif hasattr(measurable_function, "__name__"):
                name = measurable_function.__name__
            else:
                action_indices_str = "_".join(str(idx) for idx in self.action_indices)
                name = f"data_group_{action_indices_str}"

        # Record action_indices before the callable is called
        action_indices = self.action_indices

        results = measurable_function(**kwargs)

        if self.action_indices != action_indices:
            # Measurements have been performed in this function, don't measure anymore
            return

        # Ensure measuring callable matches the current action_indices
        self._verify_action(action=measurable_function, name=name, add_if_new=True)

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

    def _measure_array(
        self,
        array: Union[list, np.ndarray],
        name: str,
        label: str = None,
        unit: str = None,
        setpoints: 'Sweep' = None
    ):
        # Determine 
        ndim = np.ndim(array)

        setpoints_list = []

        # Ensure setpoints is a Sweep
        if setpoints is None:
            # Create setpoints for each dimension
            for dim, num in enumerate(np.shape(array)):
                sweep = Sweep(
                    range(num), 
                    name='setpoint_idx' + (f'_{dim}' if np.ndim(array) > 1 else ''), 
                    label='Setpoint index' + (f' dim_{dim}' if np.ndim(array) > 1 else '')
                )
                setpoints_list.append(sweep)
        elif isinstance(setpoints, Sweep):
            # Setpoints is a single Sweep
            assert ndim == 1
            assert len(setpoints) == len(array)
            setpoints_list = [setpoints]
        elif isinstance(setpoints, (list, np.ndarray)):
            if isinstance(setpoints[0], Sweep):
                setpoints_list = setpoints
            else:
                # Convert sequence to Sweep
                setpoints_list = [Sweep(setpoints, name='setpoint_idx', label='Setpoint index')]
        else:
            raise SyntaxError('Cannot measure because array setpoints not understood')

        # Enter sweep
        for setpoints in setpoints_list:
            iter(setpoints)

        # Ensure measuring array matches the current action_indices
        self._verify_action(action=None, name=name, add_if_new=True)

        self.data_handler.add_measurement_result(
            action_indices=self.action_indices,
            result=array,
            parameter=None,
            name=name,
            label=label,
            unit=unit,
        )

        for setpoints in reversed(setpoints_list):
            setpoints.exit_sweep()
        
    def _measure_dict(self, value: dict, name: str) -> Dict[str, Any]:
        """Store dictionary results

        Each key is an array name, and the value is the value to store

        Args:
            value: dictionary with (str, value) entries.
                Each element is a separate dataset array
                name: Dataset name used for dictionary
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

    def _measure_value(
        self,
        value: Union[float, int, bool],
        name: str,
        parameter: Optional[_BaseParameter] = None,
        label: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> Union[float, int, bool]:
        """Store a single value (float/int/bool)

        If this value comes from another parameter acquisition, e.g. from a
        MultiParameter, the parameter can be passed to use the right set arrays.

        Args:
            value: Value to be stored
            name: Name used for storage
            parameter: optional parameter that is passed on to
                `MeasurementLoop.measure` as a kwarg, in which case it's used
                for name, label, etc.
            label: Optional label for dat array
            unit: Optional unit for data array
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

        self.data_handler.add_measurement_result(
            action_indices=self.action_indices,
            result=value,
            parameter=parameter,
            name=name,
            label=label,
            unit=unit,
        )
        return value

    def measure(
        self,
        measurable: Union[
            Parameter, Callable, dict, float, int, bool, np.ndarray, None
        ],
        name: Optional[str] = None,
        *,  # Everything after here must be a kwarg
        label: Optional[str] = None,
        unit: Optional[str] = None,
        setpoints: Optional[Union['Sweep', Sequence]] = None,
        timestamp: bool = False,
        **kwargs,
    ) -> Any:
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
            setpoints: Optional setpoints if measuring an array, can be sequence or Sweep
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
        if self.is_stopped:
            raise SystemExit("Measurement.stop() has been called")
        if threading.current_thread() is not MeasurementLoop.measurement_thread:
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

        # Optionally record timestamp before measurement has been recorded
        if timestamp:
            t_now = datetime.now()

            # Store time referenced to t_start
            self.measure(
                (t_now - self._t_start).total_seconds(),
                "T_pre",
                unit="s",
                timestamp=False,
            )
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
        elif isinstance(measurable, (list, np.ndarray)):
            result = self._measure_array(measurable, name=name, setpoints=setpoints)
        elif isinstance(measurable, RAW_VALUE_TYPES):
            result = self._measure_value(
                measurable, name=name, label=label, unit=unit, **kwargs
            )
            self.skip()  # Increment last action index by 1
        else:
            raise RuntimeError(
                f"Cannot measure {measurable} as it cannot be called, and it "
                f"is not a dict, int, float, bool, or numpy array."
            )

        # Optionally show progress bar
        if self.show_progress:
            try:
                self._update_progress_bar(
                    action_indices=initial_action_indices,
                    description=f'Measuring {self.action_names.get(initial_action_indices)}',
                    create_if_new=True
                )
            except Exception as e:
                warn(f'Failed to update progress bar. Error: {e}')

        # Optionally record timestamp after measurement has been recorded
        if timestamp:
            t_now = datetime.now()

            # Store time referenced to t_start
            self.measure(
                (t_now - self._t_start).total_seconds(),
                "T_post",
                unit="s",
                timestamp=False,
            )
            self.skip()  # Increment last action index by 1

        self.timings.record(
            ["measurement", initial_action_indices, "total"], perf_counter() - t0
        )

        return result

    # Methods related to masking of parameters/attributes/keys
    def _mask_attr(self, obj: object, attr: str, value) -> Any:
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
                "unmask_type": "attr",
                "obj": obj,
                "attr": attr,
                "original_value": original_value,
                "value": value,
            }
        )

        return original_value

    def _mask_parameter(self, param: _BaseParameter, value: Any) -> Any:
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
                "unmask_type": "parameter",
                "obj": param,
                "original_value": original_value,
                "value": value,
            }
        )

        return original_value

    def _mask_key(self, obj: dict, key: str, value: Any) -> Any:
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
                "unmask_type": "key",
                "obj": obj,
                "key": key,
                "original_value": original_value,
                "value": value,
            }
        )

        return original_value

    def mask(self, obj: Union[object, dict], val: Any = None, **kwargs) -> Any:
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
            List of original values before masking, or single value if parameter is passed

        Examples:
            ```
            node = ParameterNode()
            node.p1 = Parameter(initial_value=1, set_cmd=None)

            with Measurement("test_masking") as msmt:
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
        obj: Union[_BaseParameter, object, dict],
        attr: Optional[str] = None,
        key: Optional[str] = None,
        unmask_type: Optional[str] = None,
        value: Optional[Any] = None,
        raise_exception: bool = True,
        remove_from_list: bool = True,
        **kwargs,  # Add kwargs because original_value may be None
    ) -> None:
        """Unmasks a previously masked object, i.e. revert value back to original

        Args:
            obj: Parameter/object/dictionary for which to revert attribute/key
            attr: object attribute to revert
            key: dictionary key to revert
            type: can be 'key', 'attr', 'parameter' if not explicitly provided by kwarg
            value: Optional masked value, only used for logging
            raise_exception: Whether to raise exception if unmasking fails
            remove_from_list: Whether to remove the masked property from the list
                msmt._masked_properties. This ensures we don't unmask twice.
        """
        if "original_value" not in kwargs:
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

            if remove_from_list:
                self._masked_properties = remaining_masked_properties
        else:
            # A masked property has been passed, which we unmask here
            try:
                original_value = kwargs["original_value"]
                if unmask_type is None:
                    if isinstance(obj, Parameter):
                        unmask_type = "parameter"
                    elif isinstance(obj, dict):
                        unmask_type = "key"
                    elif hasattr(obj, attr):
                        unmask_type = "attr"

                if unmask_type == "key":
                    obj[key] = original_value
                elif unmask_type == "attr":
                    setattr(obj, attr, original_value)
                elif unmask_type == "parameter":
                    obj(original_value)
                else:
                    raise SyntaxError(f"Unmask type {unmask_type} not understood")

                # Try to find masked property and remove from list
                if remove_from_list:
                    for masked_property in reversed(self._masked_properties):
                        if masked_property["obj"] != obj:
                            continue
                        elif attr is not None and masked_property.get("attr") != attr:
                            continue
                        elif key is not None and masked_property.get("key") != key:
                            continue
                        else:
                            self._masked_properties.remove(masked_property)
                            break

            except Exception as e:
                self.log(
                    f"Could not unmask {obj} {unmask_type} from masked value {value} "
                    f"to original value {original_value}\n"
                    f"{traceback.format_exc()}",
                    level="error",
                )

                if raise_exception:
                    raise e

    def unmask_all(self) -> None:
        """Unmask all masked properties"""
        masked_properties = reversed(self._masked_properties)
        for masked_property in masked_properties:
            self.unmask(**masked_property, raise_exception=False)
        self._masked_properties.clear()

    # Functions relating to measurement flow
    def pause(self) -> None:
        """Pause measurement at start of next parameter sweep/measurement"""
        running_measurement().is_paused = True

    def resume(self) -> None:
        """Resume measurement after being paused"""
        running_measurement().is_paused = False

    def stop(self) -> None:
        """Stop measurement at start of next parameter sweep/measurement"""
        running_measurement().is_stopped = True
        # Unpause loop
        running_measurement().resume()

    def skip(self, N: int = 1) -> Tuple[int]:
        """Skip an action index.

        Useful if a measure is only sometimes run

        Args:
            N: number of action indices to skip

        Returns:
            Measurement action_indices after skipping

        Examples:
            This measurement repeatedly creates a random value.
            It then stores the value twice, but the first time the value is
            only stored if it is above a threshold. Notice that if the random
            value is not above this threshold, the second measurement would
            become the first measurement if msmt.skip is not called
            ```
            with Measurement("skip_measurement") as msmt:
                for k in Sweep(range(10)):
                    random_value = np.random.rand()
                    if random_value > 0.7:
                        msmt.measure(random_value, "random_value_conditional")
                    else:
                        msmt.skip()

                    msmt.measure(random_value, "random_value_unconditional)
            ```
        """
        if running_measurement() is not self:
            return running_measurement().skip(N=N)
        else:
            action_indices = list(self.action_indices)
            action_indices[-1] += N
            self.action_indices = tuple(action_indices)
            return self.action_indices

    def step_out(self, reduce_dimension: bool = True) -> None:
        """Step out of a Sweep

        This function usually doesn't need to be called.
        """
        if MeasurementLoop.running_measurement is not self:
            MeasurementLoop.running_measurement.step_out(
                reduce_dimension=reduce_dimension
            )
        else:
            if reduce_dimension:
                self.loop_shape = self.loop_shape[:-1]
                self.loop_indices = self.loop_indices[:-1]

            # Remove last action index and increment one before that by one
            action_indices = list(self.action_indices[:-1])
            action_indices[-1] += 1
            self.action_indices = tuple(action_indices)

    def traceback(self) -> None:
        """Print traceback if an error occurred.

        Measurement must be ran from separate thread
        """
        if self.measurement_thread is None:
            raise RuntimeError("Measurement was not started in separate thread")

        self.measurement_thread.traceback()


def running_measurement() -> MeasurementLoop:
    """Return the running measurement"""
    return MeasurementLoop.running_measurement


class _IterateDondSweep:
    """Class used to encapsulate  `AbstractSweep` into `Sweep` as a `Sweep.sequence`"""

    def __init__(self, sweep: AbstractSweep):
        self.sweep: AbstractSweep = sweep
        self.iterator: Iterable = None
        self.parameter: _BaseParameter = sweep._param

    def __len__(self) -> int:
        return self.sweep.num_points

    def __iter__(self) -> Iterable:
        self.iterator = iter(self.sweep.get_setpoints())
        return self

    def __next__(self) -> float:
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
        label: Label of sweep. Not needed if a Parameter is passed
        unit: unit of sweep. Not needed if a Parameter is passed
        parameter: Optional parameter that is being swept over.
            If provided, the parameter value will be updated every
            time the sweep is looped over
        revert: Stores the state of a parameter before sweeping it,
            then reverts the original value upon exiting the loop.
        delay: Wait time after setting value (default zero).
        initial_delay: Delay directly after the first element.

    Examples:
        ```
        with Measurement("sweep_msmt") as msmt:
            for value in Sweep(np.linspace(5), "sweep_values"):
                msmt.measure(value, "linearly_increasing_value")

            p = Parameter("my_parameter")
            for param_val in Sweep(p.
        ```
    """
    plot_function = None

    def __init__(
        self,
        sequence: Union[Iterable, SweepValues, AbstractSweep],
        name: Optional[str] = None,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        parameter: Optional[_BaseParameter] = None,
        revert: bool = False,
        delay: Optional[float] = None,
        initial_delay: Optional[float] = None,
    ):
        if isinstance(sequence, AbstractSweep):
            sequence = _IterateDondSweep(sequence)
        elif not isinstance(sequence, Iterable):
            raise SyntaxError(f"Sweep sequence must be iterable, not {type(sequence)}")

        # Properties for the data array
        self.name: Optional[str] = name
        self.label: Optional[str] = label
        self.unit: Optional[str] = unit
        self.parameter: _BaseParameter = parameter

        self.sequence: Union[Iterable, SweepValues, AbstractSweep] = sequence
        self.dimension: Optional[int] = None
        self.loop_index: Optional[Tuple[int]] = None
        self.iterator: Optional[Iterable] = None
        self.revert: bool = revert
        self._delay: Optional[float] = delay
        self.initial_delay: Optional[float] = initial_delay

        # setpoint_info will be populated once the sweep starts
        self.setpoint_info: Optional[Dict[str, Any]] = None

        # Validate values
        if self.parameter is not None and hasattr(self.parameter, "validate"):
            for value in self.sequence:
                self.parameter.validate(value)

    def __repr__(self) -> str:
        components = []

        # Add parameter or name
        if self.parameter is not None:
            components.append(f"parameter={self.parameter}")
        elif self.name is not None:
            components.append(f"{self.name}")

        # Add number of elements
        num_elems = str(len(self.sequence)) if self.sequence is not None else "unknown"
        components.append(f"length={num_elems}")

        # Combine components
        components_str = ", ".join(components)
        return f"Sweep({components_str})"

    def __len__(self) -> int:
        return len(self.sequence)

    def __iter__(self) -> Iterable:
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
            elif self.parameter is not None:
                msmt.mask(self.parameter, self.parameter.get())
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

    def __next__(self) -> Any:
        msmt = running_measurement()

        if not msmt.is_context_manager:
            raise RuntimeError(
                "Must use the Measurement as a context manager, "
                "i.e. 'with Measurement(name) as msmt:'"
            )

        if msmt.is_stopped:
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
            self.exit_sweep()
            raise StopIteration

        # Set parameter if passed along
        if self.parameter is not None and self.parameter.settable:
            self.parameter(sweep_value)

        # Optional wait after settings value
        if self.initial_delay and self.loop_index == 0:
            sleep(self.initial_delay)
        if self.delay:
            sleep(self.delay)

        self.setpoint_info["latest_value"] = sweep_value

        self.loop_index += 1

        return sweep_value

    def __call__(
        self,
        *args: Optional[Iterable["BaseSweep"]],
        name: str = None,
        measure_params: Union[Iterable, _BaseParameter] = None,
        repetitions: int = 1,
        sweep: Union[Iterable, "BaseSweep"] = None,
        plot: bool = False,
    ):
        """Perform sweep, identical to `Sweep.execute`


        Args:
            *args: Optional additional sweeps used for N-dimensional measurements
                The first arg is the outermost sweep dimension, and the sweep on which
                `Sweep.execute` was called is the innermost dimension.
            name: Dataset name, defaults to a concatenation of sweep parameter names
            measure_params: Parameters to measure.
                If not provided, it will check the attribute ``Station.measure_params``
                for parameters. Raises an error if undefined.
            repetitions: Number of times to repeat measurement, defaults to 1.
                This will be the outermost loop if set to a value above 1.
            sweep: Identical to passing *args.
                Note that ``sweep`` can be either a single Sweep, or a Sweep list.

        Returns:
            Dataset corresponding to measurement
        """
        return self.execute(
            *args,
            name=name,
            measure_params=measure_params,
            repetitions=repetitions,
            sweep=sweep,
            plot=plot
        )

    def initialize(self) -> Dict[str, Any]:
        """Initializes a `Sweep`, attaching it to the current `MeasurementLoop`"""
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
                    name=self.name, label=self.label, unit=self.unit
                )

        setpoint_info = {
            "sweep": self,
            "parameter": self.parameter,
            "latest_value": None,
            "registered": False,
        }

        # Add to setpoint list
        msmt.setpoint_list[msmt.action_indices] = setpoint_info

        # Add to measurement actions
        assert msmt.action_indices not in msmt.actions
        msmt.actions[msmt.action_indices] = self

        return setpoint_info

    def exit_sweep(self) -> None:
        """Exits sweep, stepping out of the current `Measurement.action_indices`"""
        msmt = running_measurement()
        if self.revert:
            if isinstance(self.sequence, SweepValues):
                msmt.unmask(self.sequence.parameter)
            elif self.parameter is not None:
                msmt.unmask(self.parameter)
        msmt.step_out(reduce_dimension=True)

    def execute(
        self,
        *args: Optional[Iterable["BaseSweep"]],
        name: str = None,
        measure_params: Union[Iterable, _BaseParameter] = None,
        repetitions: int = 1,
        sweep: Union[Iterable, "BaseSweep"] = None,
        plot: bool = False,
    ) -> DataSetProtocol:
        """Performs a measurement using this sweep

        Args:
            *args: Optional additional sweeps used for N-dimensional measurements
                The first arg is the outermost sweep dimension, and the sweep on which
                `Sweep.execute` was called is the innermost dimension.
            name: Dataset name, defaults to a concatenation of sweep parameter names
            measure_params: Parameters to measure.
                If not provided, it will check the attribute ``Station.measure_params``
                for parameters. Raises an error if undefined.
            repetitions: Number of times to repeat measurement, defaults to 1.
                This will be the outermost loop if set to a value above 1.
            sweep: Identical to passing *args.
                Note that ``sweep`` can be either a single Sweep, or a Sweep list.

        Returns:
            Dataset corresponding to measurement
        """
        # Get "measure_params" from station if not provided
        if measure_params is None:
            station = Station.default
            if station is None or not getattr(station, "measure_params", None):
                raise RuntimeError(
                    "Cannot determine parameters to measure. "
                    "Either provide measure_params, or set station.measure_params"
                )
            measure_params = station.measure_params

        # Convert measure_params to list if it is a single param
        if isinstance(measure_params, _BaseParameter):
            measure_params = [measure_params]

        # Create list of sweeps
        sweeps = list(args)
        if isinstance(sweep, BaseSweep):
            sweeps.append(sweep)
        elif isinstance(sweep, (list, tuple)):
            sweeps.extend(sweep)

        if not all(isinstance(sweep, BaseSweep) for sweep in sweeps):
            raise ValueError("Args passed to Sweep.execute must be Sweeps")

        # Add repetition as a sweep if > 1
        if repetitions > 1:
            repetition_sweep = BaseSweep(range(repetitions), name="repetition")
            sweeps = [repetition_sweep] + sweeps

        # Add self as innermost sweep
        sweeps += [self]

        # Determine "name" if not provided from sweeps
        if name is None:
            dimensionality = 1 + len(sweeps)
            sweep_names = [str(sweep.name) for sweep in sweeps] + [str(self.name)]
            name = f"{dimensionality}D_sweep_" + "_".join(sweep_names)

        with MeasurementLoop(name) as msmt:
            measure_sweeps(sweeps=sweeps, measure_params=measure_params, msmt=msmt)

        if plot and Sweep.plot_function is not None and MeasurementLoop.running_measurement is None:
            Sweep.plot_function(msmt.dataset)
            plt.show()

        return msmt.dataset

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
    """Default class to create a sweep in `do1d`, `do2d`, `dond` and `MeasurementLoop`

    A Sweep can be created through its kwargs (listed below). For the most frequent
    use-cases, a Sweep can also be created by passing args in a variety of ways:

    1 arg:
    - Sweep([1,2,3], name="name")
        : sweep over sequence [1,2,3] with sweep array name "name"
        Note that kwarg "name" must be provided
    - Sweep(parameter, stop=stop_val)
        : sweep "parameter" from current value to "stop_val"
    - Sweep(parameter, around=around_val)
        : sweep "parameter" around current value with range "around_val"
        : Note that this will set ``revert`` to True if not explicitly False
    2 args:
    - Sweep(parameter, [1,2,3])
        : sweep "parameter" over sequence [1,2,3]
    - Sweep([1,2,3], "name")
        : sweep over sequence [1,2,3] with sweep array name "name"
    3 args:
    - Sweep(parameter, start_val, stop_val)
        : sweep "parameter" from "start_val" to "stop_val"
        If "num" or "step" is not given as kwarg, it will check if "num" or "step"
        is set in dict "parameter.sweep_defaults" and use that, or else raise an error.
    4 args:
    - Sweep(parameter, start_val, stop_val, num)
        : Sweep "parameter" from "start_val" to "stop_val" with "num" number of points

    Args:
        start: start value of sweep sequence
            Cannot be used together with ``around``
        stop: stop value of sweep sequence
            Cannot be used together with ``around``
        around: sweep around the current parameter value.
            ``start`` and ``stop`` are defined from ``around`` and the current value
            i.e. start=X-dx, stop=X+dx when current_value=X and around=dx.
            Passing the kwarg "around" also sets revert=True unless explicitly set False
        num: Number of points between start and stop.
            Cannot be used together with ``step``
        step: Increment from start to stop.
            Cannot be used together with ``num``
        delay: Time delay after incrementing to the next value
        initial_delay: Time delay after having incremented to its first value
        name: Sweep name, overrides parameter.name
        label: Sweep label, overrides parameter.label
        unit: Sweep unit, overrides parameter.unit
        revert: Revert parameter back to original value after the sweep ends.
            This is False by default, unless the kwarg ``around`` is passed
    """

    sequence_keywords = [
        "start",
        "stop",
        "around",
        "num",
        "step",
        "parameter",
        "sequence",
    ]
    base_keywords = [
        "delay",
        "initial_delay",
        "name",
        "label",
        "unit",
        "revert",
        "parameter",
    ]

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
        revert: bool = None,
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
            revert=revert,
        )

        sequence_kwargs, base_kwargs = self._transform_args_to_kwargs(*args, **kwargs)

        self.sequence: Iterable = self._generate_sequence(**sequence_kwargs)

        super().__init__(sequence=self.sequence, **base_kwargs)

    def _transform_args_to_kwargs(self, *args, **kwargs) -> Tuple[dict]:
        """Transforms sweep initialization args to kwargs.
        Allowed args are:

        1 arg:
        - Sweep([1,2,3], name="name")
          : sweep over sequence [1,2,3] with sweep array name "name"
          Note that kwarg "name" must be provided
        - Sweep(parameter, stop=stop_val)
          : sweep "parameter" from current value to "stop_val"
        - Sweep(parameter, around=around_val)
          : sweep "parameter" around current value with range "around_val"
          : Note that this will set ``revert`` to True if not explicitly False
        2 args:
        - Sweep(parameter, [1,2,3])
          : sweep "parameter" over sequence [1,2,3]
        - Sweep([1,2,3], "name")
          : sweep over sequence [1,2,3] with sweep array name "name"
        3 args:
        - Sweep(parameter, start_val, stop_val)
          : sweep "parameter" from "start_val" to "stop_val"
          If "num" or "step" is not given as kwarg, it will check if "num" or "step"
          if set in dict "parameter.sweep_defaults" and use that, or raise an error otherwise.
        4 args:
        - Sweep(parameter, start_val, stop_val, num)
          : Sweep "parameter" from "start_val" to "stop_val" with "num" number of points
        """
        if len(args) == 1:  # Sweep([1,2,3], name="name")
            if isinstance(args[0], Iterable):
                if kwargs.get("name") is None:
                    kwargs["name"] = "iteration"
                if kwargs.get("label") is None:
                    kwargs["label"] = "Iteration"
                (kwargs["sequence"],) = args
            elif isinstance(args[0], _BaseParameter):
                assert (
                    kwargs.get("stop") is not None or kwargs.get("around") is not None
                ), "Must provide stop value for parameter"
                (kwargs["parameter"],) = args
            elif isinstance(args[0], AbstractSweep):
                kwargs["sequence"] = _IterateDondSweep(args[0])
                parameter = kwargs["sequence"].parameter
                kwargs["name"] = kwargs["name"] or parameter.name
                kwargs["label"] = kwargs["label"] or parameter.label
                kwargs["unit"] = kwargs["unit"] or parameter.unit
            else:
                raise SyntaxError(
                    "Sweep with 1 arg must have iterable or parameter as arg"
                )
        elif len(args) == 2:
            if isinstance(args[0], _BaseParameter):  # Sweep(parameter, [1,2,3])
                if isinstance(args[1], Iterable):
                    kwargs["parameter"], kwargs["sequence"] = args
                else:
                    raise SyntaxError(
                        "Sweep with Parameter arg and second arg should have second arg"
                        " be a sequence"
                    )
            elif isinstance(args[0], Iterable):  # Sweep([1,2,3], "name")
                assert isinstance(args[1], str)
                assert kwargs.get("name") is None
                kwargs["sequence"], kwargs["name"] = args
            else:
                raise SyntaxError(
                    "Unknown sweep syntax. Either use 'Sweep(parameter, sequence)' or "
                    "'Sweep(sequence, name)'"
                )
        elif len(args) == 3:  # Sweep(parameter, 0, 1)
            assert isinstance(args[0], _BaseParameter)
            assert isinstance(args[1], (float, int))
            assert isinstance(args[2], (float, int))
            assert kwargs.get("start") is None
            assert kwargs.get("stop") is None
            kwargs["parameter"], kwargs["start"], kwargs["stop"] = args
        elif len(args) == 4:  # Sweep(parameter, 0, 1, 151)
            assert isinstance(args[0], _BaseParameter)
            assert isinstance(args[1], (float, int))
            assert isinstance(args[2], (float, int))
            assert isinstance(args[3], (float, int))
            assert kwargs.get("start") is None
            assert kwargs.get("stop") is None
            assert kwargs.get("num") is None
            kwargs["parameter"], kwargs["start"], kwargs["stop"], kwargs["num"] = args

        # Use parameter name, label, and unit if not explicitly provided
        if kwargs.get("parameter") is not None:
            kwargs.setdefault("name", kwargs["parameter"].name)
            kwargs.setdefault("label", kwargs["parameter"].label)
            kwargs.setdefault("unit", kwargs["parameter"].unit)

            # Update kwargs with sweep_defaults from parameter
            if hasattr(kwargs["parameter"], "sweep_defaults"):
                for key, val in kwargs["parameter"].sweep_defaults.items():
                    if key == 'num' and kwargs.get('step') is not None:
                        continue
                    if kwargs.get(key) is None:
                        kwargs[key] = val

        # Revert parameter to original value if kwarg "around" is passed
        # and "revert" is not explicitly False
        if kwargs["around"] is not None and kwargs["revert"] is None:
            kwargs["revert"] = True

        sequence_kwargs = {key: kwargs.get(key) for key in self.sequence_keywords}
        base_kwargs = {key: kwargs.get(key) for key in self.base_keywords}

        return sequence_kwargs, base_kwargs

    def _generate_sequence(
        self,
        start: Optional[float] = None,
        stop: Optional[float] = None,
        around: Optional[float] = None,
        num: Optional[int] = None,
        step: Optional[float] = None,
        parameter: Optional[_BaseParameter] = None,
        sequence: Optional[Iterable] = None,
    ) -> Sequence:
        """Creates a sequence from passed values"""
        # Return "sequence" if explicitly provided
        if sequence is not None:
            return sequence

        # Verify that "around" is used with "parameter" but not with "start" and "stop"
        if around is not None:
            if start is not None or stop is not None:
                raise SyntaxError(
                    "Cannot pass kwarg 'around' and also 'start' or 'stop'"
                )
            elif parameter is None:
                raise SyntaxError("Cannot use kwarg 'around' without a parameter")

            # Convert "around" to "start" and "stop" using parameter current value
            center_value = parameter()
            if center_value is None:
                raise ValueError(
                    "Parameter must have initial value if 'around' keyword is used"
                )
            start = center_value - around
            stop = center_value + around
        elif stop is not None:
            # Use "parameter" current value if "start" is not provided
            if start is None:
                if parameter is None:
                    raise SyntaxError(
                        "Cannot use 'stop' without 'start' or a 'parameter'"
                    )
                start = parameter()
                if start is None:
                    raise ValueError(
                        "Parameter must have initial value if start is not explicitly provided"
                    )
        else:
            raise SyntaxError("Must provide either 'around' or 'stop'")

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
            raise SyntaxError(
                "Cannot determine measurement points. "
                "Either provide 'sequence', 'step' or 'num'"
            )

        return sequence


def measure_sweeps(
    sweeps: List[BaseSweep],
    measure_params: List[_BaseParameter],
    msmt: "MeasurementLoop" = None,
) -> None:
    """Recursively iterate over Sweep objects, measuring measure_params in innermost loop

    This method is used to perform arbitrary-dimension by passing a list of sweeps,
    it can be compared to `dond`

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


class Iterate(Sweep):
    """Variant of Sweep that is used to iterate outside a MeasurementLoop"""
    def __iter__(self) -> Iterable:
        # Determine sweep parameter
        if self.parameter is None:
            if isinstance(self.sequence, _IterateDondSweep):
                # sweep is a doNd sweep that already has a parameter
                self.parameter = self.sequence.parameter
            else:
                # Need to create a parameter
                self.parameter = Parameter(
                    name=self.name, label=self.label, unit=self.unit
                )

        # We use this to revert back in the end
        self.original_value = self.parameter.get()

        self.loop_index = 0
        self.dimension = 1
        self.iterator = iter(self.sequence)

        return self

    def __next__(self) -> Any:
        try:  # Perform loop action
            sweep_value = next(self.iterator)
        except StopIteration:  # Reached end of iteration
            if self.revert:
                try:
                    self.parameter(self.original_value)
                except Exception:
                    warn(f'Could not revert {self.parameter} to {self.original_value}')
            raise StopIteration

        # Set parameter if passed along
        if self.parameter is not None and self.parameter.settable:
            self.parameter(sweep_value)

        # Optional wait after settings value
        if self.initial_delay and self.loop_index == 0:
            sleep(self.initial_delay)
        if self.delay:
            sleep(self.delay)

        self.loop_index += 1

        return sweep_value