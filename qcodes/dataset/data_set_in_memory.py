import json
import logging
import os
import time
import warnings
from collections.abc import Sized
from typing import (TYPE_CHECKING, Any, Dict, Hashable, Iterator,
                    List, Mapping, Optional, Sequence, Set, Tuple, Union, cast)

import numpy
import pandas as pd

from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpec, ParamSpecBase
from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.descriptions.versioning.converters import (new_to_old,
                                                               old_to_new)
from qcodes.dataset.descriptions.versioning.rundescribertypes import Shapes
from qcodes.dataset.descriptions.versioning.v0 import InterDependencies
from qcodes.dataset.export_config import (DataExportType, get_data_export_path,
                                          get_data_export_prefix,
                                          get_data_export_type)
from qcodes.dataset.guids import generate_guid
from qcodes.dataset.linked_datasets.links import (Link, links_to_str,
                                                  str_to_links)
from qcodes.dataset.sqlite.connection import (ConnectionPlus, atomic)
from qcodes.dataset.sqlite.database import conn_from_dbpath_or_conn
from qcodes.dataset.sqlite.queries import (
    add_meta_data, add_parameter, completed, create_run,
    get_completed_timestamp_from_run_id,
    get_experiment_name_from_experiment_id, get_guid_from_run_id,
    get_last_experiment, get_metadata, get_metadata_from_run_id,
    get_parameter_data, get_parent_dataset_links, get_run_description,
    get_run_timestamp_from_run_id, get_sample_name_from_experiment_id,
    mark_run_complete, run_exists, set_run_timestamp,
    update_parent_datasets, update_run_description)
from qcodes.dataset.sqlite.query_helpers import (VALUE, VALUES,
                                                 length,
                                                 select_one_where)
from qcodes.instrument.parameter import _BaseParameter
from qcodes.utils.deprecate import deprecate
from qcodes.utils.helpers import NumpyJSONEncoder

from .data_set import (SPECS, CompletedError, DataLengthException,
                       DataPathException, ParameterData, SpecsOrInterDeps,
                       values_type)
from .data_set_cache import DataSetCache
from .descriptions.versioning import serialization as serial

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

    from qcodes.station import Station

log = logging.getLogger(__name__)


class DataSetInMem(Sized):

    # the "persistent traits" are the attributes/properties of the DataSet
    # that are NOT tied to the representation of the DataSet in any particular
    # database
    persistent_traits = ('name', 'guid', 'number_of_results',
                         'parameters', 'paramspecs', 'exp_name', 'sample_name',
                         'completed', 'snapshot', 'run_timestamp_raw',
                         'description', 'completed_timestamp_raw', 'metadata',
                         'dependent_parameters', 'parent_dataset_links',
                         'captured_run_id', 'captured_counter')
    background_sleep_time = 1e-3

    def __init__(self, path_to_db: Optional[str] = None,
                 run_id: Optional[int] = None,
                 conn: Optional[ConnectionPlus] = None,
                 exp_id: Optional[int] = None,
                 name: Optional[str] = None,
                 specs: Optional[SpecsOrInterDeps] = None,
                 values: Optional[VALUES] = None,
                 metadata: Optional[Mapping[str, Any]] = None,
                 shapes: Optional[Shapes] = None,
                 in_memory_cache: bool = True) -> None:
        """
        Create a new :class:`.DataSet` object. The object can either hold a new run or
        an already existing run. If a ``run_id`` is provided, then an old run is
        looked up, else a new run is created.

        Args:
            path_to_db: path to the sqlite file on disk. If not provided, the
                path will be read from the config.
            run_id: provide this when loading an existing run, leave it
                as None when creating a new run
            conn: connection to the DB; if provided and ``path_to_db`` is
                provided as well, then a ``ValueError`` is raised (this is to
                prevent the possibility of providing a connection to a DB
                file that is different from ``path_to_db``)
            exp_id: the id of the experiment in which to create a new run.
                Ignored if ``run_id`` is provided.
            name: the name of the dataset. Ignored if ``run_id`` is provided.
            specs: paramspecs belonging to the dataset or an ``InterDependencies_``
                object that describes the dataset. Ignored if ``run_id``
                is provided.
            values: values to insert into the dataset. Ignored if ``run_id`` is
                provided.
            metadata: metadata to insert into the dataset. Ignored if ``run_id``
                is provided.
            shapes:
                An optional dict from names of dependent parameters to the shape
                of the data captured as a list of integers. The list is in the
                same order as the interdependencies or paramspecs provided.
                Ignored if ``run_id`` is provided.
            in_memory_cache: Should measured data be keep in memory
                and available as part of the `dataset.cache` object.

        """
        self.conn = conn_from_dbpath_or_conn(conn, path_to_db)

        self._parent_dataset_links: List[Link]
        #: In memory representation of the data in the dataset.
        self.cache: DataSetCache = DataSetCache(self)

        self._in_memory_cache = in_memory_cache
        self._export_path: Optional[str] = None

        if run_id is not None:
            if not run_exists(self.conn, run_id):
                raise ValueError(f"Run with run_id {run_id} does not exist in "
                                 f"the database")
            self._run_id = run_id
            self._completed = completed(self.conn, self.run_id)
            run_desc = self._get_run_description_from_db()
            self._rundescriber = run_desc
            self._metadata = get_metadata_from_run_id(self.conn, self.run_id)
            self._started = self.run_timestamp_raw is not None
            self._parent_dataset_links = str_to_links(
                get_parent_dataset_links(self.conn, self.run_id))
        else:
            # Actually perform all the side effects needed for the creation
            # of a new dataset. Note that a dataset is created (in the DB)
            # with no parameters; they are written to disk when the dataset
            # is marked as started
            if exp_id is None:
                exp_id = get_last_experiment(self.conn)
                if exp_id is None:  # if it's still None, then...
                    raise ValueError("No experiments found."
                                     "You can start a new one with:"
                                     " new_experiment(name, sample_name)")
            name = name or "dataset"
            # todo update to handle that no runs table is created
            _, run_id, __ = create_run(self.conn, exp_id, name,
                                       generate_guid(),
                                       parameters=None,
                                       values=values,
                                       metadata=metadata)
            # this is really the UUID (an ever increasing count in the db)
            self._run_id = run_id
            self._completed = False
            self._started = False

            if isinstance(specs, InterDependencies_):
                interdeps = specs
            elif specs is not None:
                interdeps = old_to_new(InterDependencies(*specs))
            else:
                interdeps = InterDependencies_()

            self.set_interdependencies(
                interdeps=interdeps,
                shapes=shapes)

            self._metadata = get_metadata_from_run_id(self.conn, self.run_id)
            self._parent_dataset_links = []

    def prepare(self,
                station: "Optional[Station]",
                interdeps: InterDependencies_,
                write_in_background: bool,
                shapes: Shapes = None,
                parent_datasets: Sequence[Dict[Any, Any]] = ()) -> None:
        if station:
            self.add_snapshot(json.dumps({'station': station.snapshot()},
                              cls=NumpyJSONEncoder))

        if interdeps == InterDependencies_():
            raise RuntimeError("No parameters supplied")
        else:
            self.set_interdependencies(interdeps,
                                       shapes)
        links = [Link(head=self.guid, **pdict)
                 for pdict in parent_datasets]
        self.parent_dataset_links = links
        self.mark_started(start_bg_writer=write_in_background)

    @property
    def run_id(self) -> int:
        return self._run_id

    @property
    def captured_run_id(self) -> int:
        return select_one_where(self.conn, "runs",
                                "captured_run_id", "run_id", self.run_id)

    @property
    def path_to_db(self) -> str:
        return self.conn.path_to_dbfile

    @property
    def name(self) -> str:
        return select_one_where(self.conn, "runs",
                                "name", "run_id", self.run_id)

    @property
    def table_name(self) -> str:
        return select_one_where(self.conn, "runs",
                                "result_table_name", "run_id", self.run_id)

    @property
    def guid(self) -> str:
        return get_guid_from_run_id(self.conn, self.run_id)

    @property
    def snapshot(self) -> Optional[Dict[str, Any]]:
        """Snapshot of the run as dictionary (or None)"""
        snapshot_json = self.snapshot_raw
        if snapshot_json is not None:
            return json.loads(snapshot_json)
        else:
            return None

    @property
    def snapshot_raw(self) -> Optional[str]:
        """Snapshot of the run as a JSON-formatted string (or None)"""
        return select_one_where(self.conn, "runs", "snapshot",
                                "run_id", self.run_id)

    @property
    def number_of_results(self) -> int:
        # todo how to replace
        return 1

    @property
    def counter(self) -> int:
        return select_one_where(self.conn, "runs",
                                "result_counter", "run_id", self.run_id)

    @property
    def captured_counter(self) -> int:
        return select_one_where(self.conn, "runs",
                                "captured_counter", "run_id", self.run_id)

    @property
    def parameters(self) -> str:
        if self.pristine:
            psnames = [ps.name for ps in self.description.interdeps.paramspecs]
            return ','.join(psnames)
        else:
            return select_one_where(self.conn, "runs",
                                    "parameters", "run_id", self.run_id)

    @property
    def paramspecs(self) -> Dict[str, ParamSpec]:
        return {ps.name: ps
                for ps in self.get_parameters()}

    @property
    def dependent_parameters(self) -> Tuple[ParamSpecBase, ...]:
        """
        Return all the parameters that explicitly depend on other parameters
        """
        return tuple(self._rundescriber.interdeps.dependencies.keys())

    @property
    def exp_id(self) -> int:
        return select_one_where(self.conn, "runs",
                                "exp_id", "run_id", self.run_id)

    @property
    def exp_name(self) -> str:
        return get_experiment_name_from_experiment_id(self.conn, self.exp_id)

    @property
    def sample_name(self) -> str:
        return get_sample_name_from_experiment_id(self.conn, self.exp_id)

    @property
    def run_timestamp_raw(self) -> Optional[float]:
        """
        Returns run timestamp as number of seconds since the Epoch

        The run timestamp is the moment when the measurement for this run
        started.
        """
        return get_run_timestamp_from_run_id(self.conn, self.run_id)

    @property
    def description(self) -> RunDescriber:
        return self._rundescriber

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def parent_dataset_links(self) -> List[Link]:
        """
        Return a list of Link objects. Each Link object describes a link from
        this dataset to one of its parent datasets
        """
        return self._parent_dataset_links

    @parent_dataset_links.setter
    def parent_dataset_links(self, links: List[Link]) -> None:
        """
        Assign one or more links to parent datasets to this dataset. It is an
        error to assign links to a non-pristine dataset

        Args:
            links: The links to assign to this dataset
        """
        if not self.pristine:
            raise RuntimeError('Can not set parent dataset links on a dataset '
                               'that has been started.')

        if not all(isinstance(link, Link) for link in links):
            raise ValueError('Invalid input. Did not receive a list of Links')

        for link in links:
            if link.head != self.guid:
                raise ValueError(
                    'Invalid input. All links must point to this dataset. '
                    'Got link(s) with head(s) pointing to another dataset.')

        self._parent_dataset_links = links

    def run_timestamp(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> Optional[str]:
        """
        Returns run timestamp in a human-readable format

        The run timestamp is the moment when the measurement for this run
        started. If the run has not yet been started, this function returns
        None.

        Consult with :func:`time.strftime` for information about the format.
        """
        if self.run_timestamp_raw is None:
            return None
        else:
            return time.strftime(fmt, time.localtime(self.run_timestamp_raw))

    @property
    def completed_timestamp_raw(self) -> Optional[float]:
        """
        Returns timestamp when measurement run was completed
        as number of seconds since the Epoch

        If the run (or the dataset) is not completed, then returns None.
        """
        return get_completed_timestamp_from_run_id(self.conn, self.run_id)

    def completed_timestamp(self,
                            fmt: str = "%Y-%m-%d %H:%M:%S") -> Optional[str]:
        """
        Returns timestamp when measurement run was completed
        in a human-readable format

        If the run (or the dataset) is not completed, then returns None.

        Consult with ``time.strftime`` for information about the format.
        """
        completed_timestamp_raw = self.completed_timestamp_raw

        if completed_timestamp_raw:
            completed_timestamp: Optional[str] = time.strftime(
                fmt, time.localtime(completed_timestamp_raw))
        else:
            completed_timestamp = None

        return completed_timestamp

    def _get_run_description_from_db(self) -> RunDescriber:
        """
        Look up the run_description from the database
        """
        desc_str = get_run_description(self.conn, self.run_id)
        return serial.from_json_to_current(desc_str)

    def set_interdependencies(self,
                              interdeps: InterDependencies_,
                              shapes: Shapes = None) -> None:
        """
        Set the interdependencies object (which holds all added
        parameters and their relationships) of this dataset and
        optionally the shapes object that holds information about
        the shape of the data to be measured.
        """
        if not isinstance(interdeps, InterDependencies_):
            raise TypeError('Wrong input type. Expected InterDepencies_, '
                            f'got {type(interdeps)}')

        if not self.pristine:
            mssg = ('Can not set interdependencies on a DataSet that has '
                    'been started.')
            raise RuntimeError(mssg)
        self._rundescriber = RunDescriber(interdeps, shapes=shapes)

    def get_parameters(self) -> SPECS:
        old_interdeps = new_to_old(self.description.interdeps)
        return list(old_interdeps.paramspecs)

    def add_metadata(self, tag: str, metadata: Any) -> None:
        """
        Adds metadata to the :class:`.DataSet`. The metadata is stored under the
        provided tag. Note that None is not allowed as a metadata value.

        Args:
            tag: represents the key in the metadata dictionary
            metadata: actual metadata
        """

        self._metadata[tag] = metadata
        # `add_meta_data` is not atomic by itself, hence using `atomic`
        with atomic(self.conn) as conn:
            add_meta_data(conn, self.run_id, {tag: metadata})

    def add_snapshot(self, snapshot: str, overwrite: bool = False) -> None:
        """
        Adds a snapshot to this run

        Args:
            snapshot: the raw JSON dump of the snapshot
            overwrite: force overwrite an existing snapshot
        """
        if self.snapshot is None or overwrite:
            add_meta_data(self.conn, self.run_id, {'snapshot': snapshot})
        elif self.snapshot is not None and not overwrite:
            log.warning('This dataset already has a snapshot. Use overwrite'
                        '=True to overwrite that')

    @property
    def pristine(self) -> bool:
        """
        Is this :class:`.DataSet` pristine? A pristine :class:`.DataSet` has not yet been started,
        meaning that parameters can still be added and removed, but results
        can not be added.
        """
        return not(self._started or self._completed)

    @property
    def running(self) -> bool:
        """
        Is this :class:`.DataSet` currently running? A running :class:`.DataSet` has been started,
        but not yet completed.
        """
        return self._started and not self._completed

    @property
    def started(self) -> bool:
        """
        Has this :class:`.DataSet` been started? A :class:`.DataSet` not started can not have any
        results added to it.
        """
        return self._started

    @property
    def completed(self) -> bool:
        """
        Is this :class:`.DataSet` completed? A completed :class:`.DataSet` may not be modified in
        any way.
        """
        return self._completed

    @completed.setter
    def completed(self, value: bool) -> None:
        self._completed = value
        if value:
            mark_run_complete(self.conn, self.run_id)

    def mark_started(self, start_bg_writer: bool = False) -> None:
        """
        Mark this :class:`.DataSet` as started. A :class:`.DataSet` that has been started can not
        have its parameters modified.

        Calling this on an already started :class:`.DataSet` is a NOOP.

        Args:
            start_bg_writer: If True, the add_results method will write to the
                database in a separate thread.
        """
        if not self._started:
            self._perform_start_actions(start_bg_writer=start_bg_writer)
            self._started = True

    def _perform_start_actions(self, start_bg_writer: bool) -> None:
        """
        Perform the actions that must take place once the run has been started
        """
        paramspecs = new_to_old(self._rundescriber.interdeps).paramspecs

        for spec in paramspecs:
            add_parameter(self.conn, self.table_name, spec)

        desc_str = serial.to_json_for_storage(self.description)

        update_run_description(self.conn, self.run_id, desc_str)

        set_run_timestamp(self.conn, self.run_id)

        pdl_str = links_to_str(self._parent_dataset_links)
        update_parent_datasets(self.conn, self.run_id, pdl_str)

    def mark_completed(self) -> None:
        """
        Mark :class:`.DataSet` as complete and thus read only and notify the subscribers
        """
        if self.completed:
            return
        if self.pristine:
            raise RuntimeError('Can not mark DataSet as complete before it '
                               'has been marked as started.')

        self.completed = True

    def _raise_if_not_writable(self) -> None:
        if self.pristine:
            raise RuntimeError('This DataSet has not been marked as started. '
                               'Please mark the DataSet as started before '
                               'adding results to it.')
        if self.completed:
            raise CompletedError('This DataSet is complete, no further '
                                 'results can be added to it.')

    @staticmethod
    def _validate_parameters(*params: Union[str, ParamSpec, _BaseParameter]
                             ) -> List[str]:
        """
        Validate that the provided parameters have a name and return those
        names as a list.
        The Parameters may be a mix of strings, ParamSpecs or ordinary
        QCoDeS parameters.
        """

        valid_param_names = []
        for maybeParam in params:
            if isinstance(maybeParam, str):
                valid_param_names.append(maybeParam)
                continue
            else:
                try:
                    maybeParam = maybeParam.name
                except Exception as e:
                    raise ValueError(
                        "This parameter does not have  a name") from e
                valid_param_names.append(maybeParam)
        return valid_param_names

    def get_metadata(self, tag: str) -> str:
        return get_metadata(self.conn, tag, self.table_name)

    def __len__(self) -> int:
        return length(self.conn, self.table_name)

    def __repr__(self) -> str:
        out = []
        heading = f"{self.name} #{self.run_id}@{self.path_to_db}"
        out.append(heading)
        out.append("-" * len(heading))
        ps = self.get_parameters()
        if len(ps) > 0:
            for p in ps:
                out.append(f"{p.name} - {p.type}")

        return "\n".join(out)

    def _enqueue_results(
            self, result_dict: Mapping[ParamSpecBase, numpy.ndarray]) -> None:
        """
        Enqueue the results into self._results

        Before we can enqueue the results, all values of the results dict
        must have the same length. We enqueue each parameter tree separately,
        effectively mimicking making one call to add_results per parameter
        tree.

        Deal with 'numeric' type parameters. If a 'numeric' top level parameter
        has non-scalar shape, it must be unrolled into a list of dicts of
        single values (database).
        """
        self._raise_if_not_writable()
        interdeps = self._rundescriber.interdeps

        toplevel_params = (set(interdeps.dependencies)
                           .intersection(set(result_dict)))
        if self._in_memory_cache:
            new_results: Dict[str, Dict[str, numpy.ndarray]] = {}
        for toplevel_param in toplevel_params:
            inff_params = set(interdeps.inferences.get(toplevel_param, ()))
            deps_params = set(interdeps.dependencies.get(toplevel_param, ()))
            all_params = (inff_params
                          .union(deps_params)
                          .union({toplevel_param}))

            if self._in_memory_cache:
                new_results[toplevel_param.name] = {}
                new_results[toplevel_param.name][toplevel_param.name] = self._reshape_array_for_cache(
                    toplevel_param,
                    result_dict[toplevel_param]
                )
                for param in all_params:
                    if param is not toplevel_param:
                        new_results[toplevel_param.name][param.name] = self._reshape_array_for_cache(
                            param,
                            result_dict[param]
                        )

        # Finally, handle standalone parameters

        standalones = (set(interdeps.standalones)
                       .intersection(set(result_dict)))

        if standalones:
            if self._in_memory_cache:
                for st in standalones:
                    new_results[st.name] = {
                        st.name: self._reshape_array_for_cache(st, result_dict[st])
                    }

        if self._in_memory_cache:
            self.cache.add_data(new_results)

    @staticmethod
    def _reshape_array_for_cache(
            param: ParamSpecBase,
            param_data: numpy.ndarray
    ) -> numpy.ndarray:
        """
        Shape cache data so it matches data read from database.
        This means:

        - Add an extra singleton dim to array data
        - flatten non array data into a linear array.
        """
        param_data = numpy.atleast_1d(param_data)
        if param.type == "array":
            new_data = numpy.reshape(
                param_data,
                (1,) + param_data.shape
            )
        else:
            new_data = param_data.ravel()
        return new_data

    @staticmethod
    def _finalize_res_dict_array(
            result_dict: Mapping[ParamSpecBase, values_type],
            all_params: Set[ParamSpecBase]) -> List[Dict[str, VALUE]]:
        """
        Make a list of res_dicts out of the results for a 'array' type
        parameter. The results are assumed to already have been validated for
        type and shape
        """

        def reshaper(val: Any, ps: ParamSpecBase) -> VALUE:
            paramtype = ps.type
            if paramtype == 'numeric':
                return float(val)
            elif paramtype == 'text':
                return str(val)
            elif paramtype == 'complex':
                return complex(val)
            elif paramtype == 'array':
                if val.shape:
                    return val
                else:
                    return numpy.reshape(val, (1,))
            else:
                raise ValueError(f'Cannot handle unknown paramtype '
                                 f'{paramtype!r} of {ps!r}.')

        res_dict = {ps.name: reshaper(result_dict[ps], ps)
                    for ps in all_params}

        return [res_dict]

    @staticmethod
    def _finalize_res_dict_numeric_text_or_complex(
            result_dict: Mapping[ParamSpecBase, numpy.ndarray],
            toplevel_param: ParamSpecBase,
            inff_params: Set[ParamSpecBase],
            deps_params: Set[ParamSpecBase]) -> List[Dict[str, VALUE]]:
        """
        Make a res_dict in the format expected by DataSet.add_results out
        of the results for a 'numeric' or text type parameter. This includes
        replicating and unrolling values as needed and also handling the corner
        case of np.array(1) kind of values
        """

        res_list: List[Dict[str, VALUE]] = []
        all_params = inff_params.union(deps_params).union({toplevel_param})

        t_map = {'numeric': float, 'text': str, 'complex': complex}

        toplevel_shape = result_dict[toplevel_param].shape
        if toplevel_shape == ():
            # In the case of a single value, life is reasonably simple
            res_list = [{ps.name: t_map[ps.type](result_dict[ps])
                         for ps in all_params}]
        else:
            # We first massage all values into np.arrays of the same
            # shape
            flat_results: Dict[str, numpy.ndarray] = {}

            toplevel_val = result_dict[toplevel_param]
            flat_results[toplevel_param.name] = toplevel_val.ravel()
            N = len(flat_results[toplevel_param.name])
            for dep in deps_params:
                if result_dict[dep].shape == ():
                    flat_results[dep.name] = numpy.repeat(result_dict[dep], N)
                else:
                    flat_results[dep.name] = result_dict[dep].ravel()
            for inff in inff_params:
                if numpy.shape(result_dict[inff]) == ():
                    flat_results[inff.name] = numpy.repeat(result_dict[dep], N)
                else:
                    flat_results[inff.name] = result_dict[inff].ravel()

            # And then put everything into the list

            res_list = [{p.name: flat_results[p.name][ind] for p in all_params}
                        for ind in range(N)]

        return res_list

    @staticmethod
    def _finalize_res_dict_standalones(
            result_dict: Mapping[ParamSpecBase, numpy.ndarray]
    ) -> List[Dict[str, VALUE]]:
        """
        Massage all standalone parameters into the correct shape
        """
        res_list: List[Dict[str, VALUE]] = []
        for param, value in result_dict.items():
            if param.type == 'text':
                if value.shape:
                    res_list += [{param.name: str(val)} for val in value]
                else:
                    res_list += [{param.name: str(value)}]
            elif param.type == 'numeric':
                if value.shape:
                    res_list += [{param.name: number} for number in value]
                else:
                    res_list += [{param.name: float(value)}]
            elif param.type == 'complex':
                if value.shape:
                    res_list += [{param.name: number} for number in value]
                else:
                    res_list += [{param.name: complex(value)}]
            else:
                res_list += [{param.name: value}]

        return res_list

    def _flush_data_to_database(self, block: bool = False) -> None:
        pass

    def _export_file_name(self, prefix: str, export_type: DataExportType) -> str:
        """Get export file name"""
        extension = export_type.value
        return f"{prefix}{self.run_id}.{extension}"

    def _export_as_netcdf(self, path: str, file_name: str) -> str:
        """Export data as netcdf to a given path with file prefix"""
        file_path = os.path.join(path, file_name)
        xarr_dataset = self.to_xarray_dataset()
        xarr_dataset.to_netcdf(path=file_path)
        return path

    def _export_as_csv(self, path: str, file_name: str) -> str:
        """Export data as csv to a given path with file prefix"""
        self.write_data_to_text_file(path=path, single_file=True, single_file_name=file_name)
        return os.path.join(path, file_name)

    def _export_data(self,
                     export_type: DataExportType,
                     path: Optional[str] = None,
                     prefix: Optional[str] = None
                     ) -> Optional[str]:
        """Export data to disk with file name {prefix}{run_id}.{ext}.
        Values for the export type, path and prefix can also be set in the qcodes
        "dataset" config.

        Args:
            export_type: Data export type, e.g. DataExportType.NETCDF
            path: Export path, defaults to value set in config
            prefix: File prefix, e.g. "qcodes_", defaults to value set in config.

        Returns:
            str: Path file was saved to, returns None if no file was saved.
        """
        # Set defaults to values in config if the value was not set
        # (defaults to None)
        path = path if path is not None else get_data_export_path()
        prefix = prefix if prefix is not None else get_data_export_prefix()

        if DataExportType.NETCDF == export_type:
            file_name = self._export_file_name(
                prefix=prefix, export_type=DataExportType.NETCDF)
            return self._export_as_netcdf(path=path, file_name=file_name)

        elif DataExportType.CSV == export_type:
            file_name = self._export_file_name(
                prefix=prefix, export_type=DataExportType.CSV)
            return self._export_as_csv(path=path, file_name=file_name)

        else:
            return None

    def export(self,
               export_type: Optional[Union[DataExportType, str]] = None,
               path: Optional[str] = None,
               prefix: Optional[str] = None) -> None:
        """Export data to disk with file name {prefix}{run_id}.{ext}.
        Values for the export type, path and prefix can also be set in the "dataset"
        section of qcodes config.

        Args:
            export_type: Data export type, e.g. "netcdf" or ``DataExportType.NETCDF``,
                defaults to a value set in qcodes config
            path: Export path, defaults to value set in config
            prefix: File prefix, e.g. ``qcodes_``, defaults to value set in config.

        Raises:
            ValueError: If the export data type is not specified, raise an error
        """
        export_type = get_data_export_type(export_type)

        if export_type is None:
            raise ValueError(
                "No data export type specified. Please set the export data type "
                "by using ``qcodes.dataset.export_config.set_data_export_type`` or "
                "give an explicit export_type when calling ``dataset.export`` manually."
            )

        self._export_path = self._export_data(
            export_type=export_type,
            path=path,
            prefix=prefix
        )

    @property
    def export_path(self) -> Optional[str]:
        return self._export_path
