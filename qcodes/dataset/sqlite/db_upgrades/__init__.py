import logging
from collections import defaultdict
from functools import wraps
from typing import Dict, Callable, DefaultDict, List, Tuple, Sequence

import numpy as np
from tqdm import tqdm

from qcodes.dataset.guids import generate_guid
from qcodes.dataset.sqlite.connection import ConnectionPlus, \
    atomic_transaction, atomic, transaction
from qcodes.dataset.sqlite.db_upgrades.version import get_user_version, \
    set_user_version
from qcodes.dataset.sqlite.query_helpers import many_many, one, insert_column
from qcodes.dataset.param_spec import ParamSpec
from qcodes.dataset.dependencies import InterDependencies
from qcodes.dataset.descriptions import RunDescriber


log = logging.getLogger(__name__)


# INFRASTRUCTURE FOR UPGRADE FUNCTIONS


# Functions decorated as 'upgrader' are inserted into this dict
# The newest database version is thus determined by the number of upgrades
# in this module
# The key is the TARGET VERSION of the upgrade, i.e. the first key is 1
_UPGRADE_ACTIONS: Dict[int, Callable] = {}


def _latest_available_version() -> int:
    """Return latest available database schema version"""
    return len(_UPGRADE_ACTIONS)


def upgrader(func: Callable[[ConnectionPlus], None]):
    """
    Decorator for database version upgrade functions. An upgrade function
    must have the name `perform_db_upgrade_N_to_M` where N = M-1. For
    simplicity, an upgrade function must take a single argument of type
    `ConnectionPlus`. The upgrade function must either perform the upgrade
    and return (no return values allowed) or fail to perform the upgrade,
    in which case it must raise a RuntimeError. A failed upgrade must be
    completely rolled back before the RuntimeError is raises.

    The decorator takes care of logging about the upgrade and managing the
    database versioning.
    """
    name_comps = func.__name__.split('_')
    if not len(name_comps) == 6:
        raise NameError('Decorated function not a valid upgrader. '
                        'Must have name "perform_db_upgrade_N_to_M"')
    if not ''.join(name_comps[:3]+[name_comps[4]]) == 'performdbupgradeto':
        raise NameError('Decorated function not a valid upgrader. '
                        'Must have name "perform_db_upgrade_N_to_M"')
    from_version = int(name_comps[3])
    to_version = int(name_comps[5])

    if not to_version == from_version+1:
        raise ValueError(f'Invalid upgrade versions in function name: '
                         f'{func.__name__}; upgrade from version '
                         f'{from_version} to version {to_version}.'
                         ' Can only upgrade from version N'
                         ' to version N+1')

    @wraps(func)
    def do_upgrade(conn: ConnectionPlus) -> None:

        log.info(f'Starting database upgrade version {from_version} '
                 f'to {to_version}')

        start_version = get_user_version(conn)
        if start_version != from_version:
            log.info(f'Skipping upgrade {from_version} -> {to_version} as'
                     f' current database version is {start_version}.')
            return

        # This function either raises or returns
        func(conn)

        set_user_version(conn, to_version)
        log.info(f'Succesfully performed upgrade {from_version} '
                 f'-> {to_version}')

    _UPGRADE_ACTIONS[to_version] = do_upgrade

    return do_upgrade


def perform_db_upgrade(conn: ConnectionPlus, version: int = -1) -> None:
    """
    This is intended to perform all upgrades as needed to bring the
    db from version 0 to the most current version (or the version specified).
    All the perform_db_upgrade_X_to_Y functions must raise if they cannot
    upgrade and be a NOOP if the current version is higher than their target.

    Args:
        conn: object for connection to the database
        version: Which version to upgrade to. We count from 0. -1 means
          'newest version'
    """
    version = _latest_available_version() if version == -1 else version

    current_version = get_user_version(conn)
    if current_version < version:
        log.info("Commencing database upgrade")
        for target_version in sorted(_UPGRADE_ACTIONS)[:version]:
            _UPGRADE_ACTIONS[target_version](conn)


# DATABASE UPGRADE FUNCTIONS


@upgrader
def perform_db_upgrade_0_to_1(conn: ConnectionPlus) -> None:
    """
    Perform the upgrade from version 0 to version 1

    Add a GUID column to the runs table and assign guids for all existing runs
    """

    sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
    cur = atomic_transaction(conn, sql)
    n_run_tables = len(cur.fetchall())

    if n_run_tables == 1:
        with atomic(conn) as conn:
            sql = "ALTER TABLE runs ADD COLUMN guid TEXT"
            transaction(conn, sql)
            # now assign GUIDs to existing runs
            cur = transaction(conn, 'SELECT run_id FROM runs')
            run_ids = [r[0] for r in many_many(cur, 'run_id')]

            for run_id in run_ids:
                query = f"""
                        SELECT run_timestamp
                        FROM runs
                        WHERE run_id == {run_id}
                        """
                cur = transaction(conn, query)
                timestamp = one(cur, 'run_timestamp')
                timeint = int(np.round(timestamp*1000))
                sql = f"""
                        UPDATE runs
                        SET guid = ?
                        where run_id == {run_id}
                        """
                sampleint = 3736062718  # 'deafcafe'
                cur.execute(sql, (generate_guid(timeint=timeint,
                                                sampleint=sampleint),))
    else:
        raise RuntimeError(f"found {n_run_tables} runs tables expected 1")


@upgrader
def perform_db_upgrade_1_to_2(conn: ConnectionPlus) -> None:
    """
    Perform the upgrade from version 1 to version 2

    Add two indeces on the runs table, one for exp_id and one for GUID
    """

    sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
    cur = atomic_transaction(conn, sql)
    n_run_tables = len(cur.fetchall())

    if n_run_tables == 1:
        _IX_runs_exp_id = """
                          CREATE INDEX
                          IF NOT EXISTS IX_runs_exp_id
                          ON runs (exp_id DESC)
                          """
        _IX_runs_guid = """
                        CREATE INDEX
                        IF NOT EXISTS IX_runs_guid
                        ON runs (guid DESC)
                        """
        with atomic(conn) as conn:
            transaction(conn, _IX_runs_exp_id)
            transaction(conn, _IX_runs_guid)
    else:
        raise RuntimeError(f"found {n_run_tables} runs tables expected 1")


def _2to3_get_result_tables(conn: ConnectionPlus) -> Dict[int, str]:
    rst_query = "SELECT run_id, result_table_name FROM runs"
    cur = conn.cursor()
    cur.execute(rst_query)

    data = cur.fetchall()
    cur.close()
    results = {}
    for row in data:
        results[row['run_id']] = row['result_table_name']
    return results


def _2to3_get_layout_ids(conn: ConnectionPlus) -> DefaultDict[int, List[int]]:
    query = """
            select runs.run_id, layouts.layout_id
            FROM layouts
            INNER JOIN runs ON runs.run_id == layouts.run_id
            """
    cur = conn.cursor()
    cur.execute(query)
    data = cur.fetchall()
    cur.close()

    results: DefaultDict[int, List[int]] = defaultdict(list)

    for row in data:
        run_id = row['run_id']
        layout_id = row['layout_id']
        results[run_id].append(layout_id)

    return results


def _2to3_get_indeps(conn: ConnectionPlus) -> DefaultDict[int, List[int]]:
    query = """
            SELECT layouts.run_id, layouts.layout_id
            FROM layouts
            INNER JOIN dependencies
            ON layouts.layout_id==dependencies.independent
            """
    cur = conn.cursor()
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    results: DefaultDict[int, List[int]] = defaultdict(list)

    for row in data:
        run_id = row['run_id']
        layout_id = row['layout_id']
        results[run_id].append(layout_id)

    return results


def _2to3_get_deps(conn: ConnectionPlus) -> DefaultDict[int, List[int]]:
    query = """
            SELECT layouts.run_id, layouts.layout_id
            FROM layouts
            INNER JOIN dependencies
            ON layouts.layout_id==dependencies.dependent
            """
    cur = conn.cursor()
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    results: DefaultDict[int, List[int]] = defaultdict(list)

    for row in data:
        run_id = row['run_id']
        layout_id = row['layout_id']
        results[run_id].append(layout_id)

    return results


def _2to3_get_dependencies(conn: ConnectionPlus
                           ) -> DefaultDict[int, List[int]]:
    query = """
            SELECT dependent, independent
            FROM dependencies
            ORDER BY dependent, axis_num ASC
            """
    cur = conn.cursor()
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    results: DefaultDict[int, List[int]] = defaultdict(list)

    if len(data) == 0:
        return results

    for row in data:
        dep = row['dependent']
        indep = row['independent']
        results[dep].append(indep)

    return results


def _2to3_get_layouts(conn: ConnectionPlus) -> Dict[int,
                                                    Tuple[str, str, str, str]]:
    query = """
            SELECT layout_id, parameter, label, unit, inferred_from
            FROM layouts
            """
    cur = conn.cursor()
    cur.execute(query)

    results: Dict[int, Tuple[str, str, str, str]] = {}
    for row in cur.fetchall():
        results[row['layout_id']] = (row['parameter'],
                                     row['label'],
                                     row['unit'],
                                     row['inferred_from'])
    return results


def _2to3_get_paramspecs(conn: ConnectionPlus,
                         layout_ids: List[int],
                         layouts: Dict[int, Tuple[str, str, str, str]],
                         dependencies: Dict[int, List[int]],
                         deps: Sequence[int],
                         indeps: Sequence[int],
                         result_table_name: str) -> Dict[int, ParamSpec]:

    paramspecs: Dict[int, ParamSpec] = {}

    the_rest = set(layout_ids).difference(set(deps).union(set(indeps)))

    # We ensure that we first retrieve the ParamSpecs on which other ParamSpecs
    # depend, then the dependent ParamSpecs and finally the rest

    for layout_id in list(indeps) + list(deps) + list(the_rest):
        (name, label, unit, inferred_from_str) = layouts[layout_id]
        # get the data type
        sql = f'PRAGMA TABLE_INFO("{result_table_name}")'
        c = transaction(conn, sql)
        paramtype = None
        for row in c.fetchall():
            if row['name'] == name:
                paramtype = row['type']
                break
        if paramtype is None:
            raise TypeError(f"Could not determine type of {name} during the"
                            f"db upgrade of {result_table_name}")

        inferred_from: List[str] = []
        depends_on: List[str] = []

        # this parameter depends on another parameter
        if layout_id in deps:
            setpoints = dependencies[layout_id]
            depends_on = [paramspecs[idp].name for idp in setpoints]

        if inferred_from_str != '':
            inferred_from = inferred_from_str.split(', ')

        paramspec = ParamSpec(name=name,
                              paramtype=paramtype,
                              label=label, unit=unit,
                              depends_on=depends_on,
                              inferred_from=inferred_from)
        paramspecs[layout_id] = paramspec

    return paramspecs


@upgrader
def perform_db_upgrade_2_to_3(conn: ConnectionPlus) -> None:
    """
    Perform the upgrade from version 2 to version 3

    Insert a new column, run_description, to the runs table and fill it out
    for exisitng runs with information retrieved from the layouts and
    dependencies tables represented as the to_json output of a RunDescriber
    object
    """

    no_of_runs_query = "SELECT max(run_id) FROM runs"
    no_of_runs = one(atomic_transaction(conn, no_of_runs_query), 'max(run_id)')
    no_of_runs = no_of_runs or 0

    # If one run fails, we want the whole upgrade to roll back, hence the
    # entire upgrade is one atomic transaction

    with atomic(conn) as conn:
        sql = "ALTER TABLE runs ADD COLUMN run_description TEXT"
        transaction(conn, sql)

        result_tables = _2to3_get_result_tables(conn)
        layout_ids_all = _2to3_get_layout_ids(conn)
        indeps_all = _2to3_get_indeps(conn)
        deps_all = _2to3_get_deps(conn)
        layouts = _2to3_get_layouts(conn)
        dependencies = _2to3_get_dependencies(conn)

        pbar = tqdm(range(1, no_of_runs+1))
        pbar.set_description("Upgrading database")

        for run_id in pbar:

            if run_id in layout_ids_all:

                result_table_name = result_tables[run_id]
                layout_ids = list(layout_ids_all[run_id])
                if run_id in indeps_all:
                    independents = tuple(indeps_all[run_id])
                else:
                    independents = ()
                if run_id in deps_all:
                    dependents = tuple(deps_all[run_id])
                else:
                    dependents = ()

                paramspecs = _2to3_get_paramspecs(conn,
                                                  layout_ids,
                                                  layouts,
                                                  dependencies,
                                                  dependents,
                                                  independents,
                                                  result_table_name)

                interdeps = InterDependencies(*paramspecs.values())
                desc = RunDescriber(interdeps=interdeps)
                json_str = desc.to_json()

            else:

                json_str = RunDescriber(InterDependencies()).to_json()

            sql = f"""
                   UPDATE runs
                   SET run_description = ?
                   WHERE run_id == ?
                   """
            cur = conn.cursor()
            cur.execute(sql, (json_str, run_id))
            log.debug(f"Upgrade in transition, run number {run_id}: OK")


@upgrader
def perform_db_upgrade_3_to_4(conn: ConnectionPlus) -> None:
    """
    Perform the upgrade from version 3 to version 4. This really
    repeats the version 3 upgrade as it originally had two bugs in
    the inferred annotation. inferred_from was passed incorrectly
    resulting in the parameter being marked inferred_from for each char
    in the inferred_from variable and inferred_from was not handled
    correctly for parameters that were neither dependencies nor dependent on
    other parameters. Both have since been fixed so rerun the upgrade.
    """

    no_of_runs_query = "SELECT max(run_id) FROM runs"
    no_of_runs = one(atomic_transaction(conn, no_of_runs_query), 'max(run_id)')
    no_of_runs = no_of_runs or 0

    # If one run fails, we want the whole upgrade to roll back, hence the
    # entire upgrade is one atomic transaction

    with atomic(conn) as conn:

        result_tables = _2to3_get_result_tables(conn)
        layout_ids_all = _2to3_get_layout_ids(conn)
        indeps_all = _2to3_get_indeps(conn)
        deps_all = _2to3_get_deps(conn)
        layouts = _2to3_get_layouts(conn)
        dependencies = _2to3_get_dependencies(conn)

        pbar = tqdm(range(1, no_of_runs+1))
        pbar.set_description("Upgrading database")

        for run_id in pbar:

            if run_id in layout_ids_all:

                result_table_name = result_tables[run_id]
                layout_ids = list(layout_ids_all[run_id])
                if run_id in indeps_all:
                    independents = tuple(indeps_all[run_id])
                else:
                    independents = ()
                if run_id in deps_all:
                    dependents = tuple(deps_all[run_id])
                else:
                    dependents = ()

                paramspecs = _2to3_get_paramspecs(conn,
                                                  layout_ids,
                                                  layouts,
                                                  dependencies,
                                                  dependents,
                                                  independents,
                                                  result_table_name)

                interdeps = InterDependencies(*paramspecs.values())
                desc = RunDescriber(interdeps=interdeps)
                json_str = desc.to_json()

            else:

                json_str = RunDescriber(InterDependencies()).to_json()

            sql = f"""
                   UPDATE runs
                   SET run_description = ?
                   WHERE run_id == ?
                   """
            cur = conn.cursor()
            cur.execute(sql, (json_str, run_id))
            log.debug(f"Upgrade in transition, run number {run_id}: OK")


@upgrader
def perform_db_upgrade_4_to_5(conn: ConnectionPlus) -> None:
    """
    Perform the upgrade from version 4 to version 5.

    Make sure that 'snapshot' column always exists in the 'runs' table. This
    was not the case before because 'snapshot' was treated as 'metadata',
    hence the 'snapshot' column was dynamically created once there was a run
    with snapshot information.
    """
    with atomic(conn) as conn:
        insert_column(conn, 'runs', 'snapshot', 'TEXT')
