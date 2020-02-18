"""
This module provides infrastructure for upgrading schema and structure of
QCoDeS database. It also provides concreted upgrade functions between all
the database versions which exist so far.

The module contains :mod:`.version` module for working with the version of
QCoDeS databases.

If needed, this module may contain modules with upgrade-specific code. The
intention is to be able to decouple the code of the upgrade functions from
the current state of the SQLite API in QCoDeS (:mod:`.sqlite`). In
principle, the upgrade functions should not have dependecies from
:mod:`.queries` module.
"""
import logging
from functools import wraps
from typing import Dict, Callable
import sys

import numpy as np
from tqdm import tqdm

from qcodes.dataset.guids import generate_guid
from qcodes.dataset.sqlite.connection import ConnectionPlus, \
    atomic_transaction, atomic, transaction
from qcodes.dataset.sqlite.db_upgrades.version import get_user_version, \
    set_user_version
from qcodes.dataset.sqlite.query_helpers import many_many, one, insert_column


log = logging.getLogger(__name__)


# INFRASTRUCTURE FOR UPGRADE FUNCTIONS


TUpgraderFunction = Callable[[ConnectionPlus], None]

# Functions decorated as 'upgrader' are inserted into this dict
# The newest database version is thus determined by the number of upgrades
# in this module
# The key is the TARGET VERSION of the upgrade, i.e. the first key is 1
_UPGRADE_ACTIONS: Dict[int, Callable] = {}


def _latest_available_version() -> int:
    """Return latest available database schema version"""
    return len(_UPGRADE_ACTIONS)


def upgrader(func: TUpgraderFunction) -> TUpgraderFunction:
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

            pbar = tqdm(range(1, len(run_ids) + 1), file=sys.stdout)
            pbar.set_description("Upgrading database; v0 -> v1")

            for run_id in pbar:
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

    pbar = tqdm(range(1), file=sys.stdout)
    pbar.set_description("Upgrading database; v1 -> v2")

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
            # iterate through the pbar for the sake of the side effect; it
            # prints that the database is being upgraded
            for _ in pbar:
                transaction(conn, _IX_runs_exp_id)
                transaction(conn, _IX_runs_guid)
    else:
        raise RuntimeError(f"found {n_run_tables} runs tables expected 1")


@upgrader
def perform_db_upgrade_2_to_3(conn: ConnectionPlus) -> None:
    """
    Perform the upgrade from version 2 to version 3

    Insert a new column, run_description, to the runs table and fill it out
    for exisitng runs with information retrieved from the layouts and
    dependencies tables represented as the json output of a RunDescriber
    object
    """
    from qcodes.dataset.sqlite.db_upgrades.upgrade_2_to_3 import upgrade_2_to_3
    upgrade_2_to_3(conn)


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
    from qcodes.dataset.sqlite.db_upgrades.upgrade_3_to_4 import upgrade_3_to_4
    upgrade_3_to_4(conn)


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
        pbar = tqdm(range(1), file=sys.stdout)
        pbar.set_description("Upgrading database; v4 -> v5")
        # iterate through the pbar for the sake of the side effect; it
        # prints that the database is being upgraded
        for _ in pbar:
            insert_column(conn, 'runs', 'snapshot', 'TEXT')


@upgrader
def perform_db_upgrade_5_to_6(conn: ConnectionPlus) -> None:
    """
    Perform the upgrade from version 5 to version 6.

    The upgrade ensures that the runs_description has a top-level entry
    called 'version'. Note that version changes of the runs_description will
    not be tracked as schema upgrades.
    """
    from qcodes.dataset.sqlite.db_upgrades.upgrade_5_to_6 import upgrade_5_to_6
    upgrade_5_to_6(conn)


@upgrader
def perform_db_upgrade_6_to_7(conn: ConnectionPlus) -> None:
    """
    Perform the upgrade from version 6 to version 7

    Add a captured_run_id and captured_counter column to the runs table and
    assign the value from the run_id and result_counter to these columns.
    """

    sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
    cur = atomic_transaction(conn, sql)
    n_run_tables = len(cur.fetchall())

    if n_run_tables == 1:

        pbar = tqdm(range(1), file=sys.stdout)
        pbar.set_description("Upgrading database; v6 -> v7")
        # iterate through the pbar for the sake of the side effect; it
        # prints that the database is being upgraded
        for _ in pbar:
            with atomic(conn) as conn:
                sql = "ALTER TABLE runs ADD COLUMN captured_run_id"
                transaction(conn, sql)
                sql = "ALTER TABLE runs ADD COLUMN captured_counter"
                transaction(conn, sql)

                sql = f"""
                        UPDATE runs
                        SET captured_run_id = run_id,
                            captured_counter = result_counter
                        """
                transaction(conn, sql)
    else:
        raise RuntimeError(f"found {n_run_tables} runs tables expected 1")


@upgrader
def perform_db_upgrade_7_to_8(conn: ConnectionPlus) -> None:
    """
    Perform the upgrade from version 7 to version 8.

    Add a new column to store the dataset's parents to the runs table.
    """
    with atomic(conn) as conn:
        pbar = tqdm(range(1), file=sys.stdout)
        pbar.set_description("Upgrading database; v7 -> v8")
        # iterate through the pbar for the sake of the side effect; it
        # prints that the database is being upgraded
        for _ in pbar:
            insert_column(conn, 'runs', 'parent_datasets', 'TEXT')


@upgrader
def perform_db_upgrade_8_to_9(conn: ConnectionPlus) -> None:
    """
    Perform the upgrade from version 8 to version 9.

    Add indices on the runs table for captured_run_id
    """

    sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
    cur = atomic_transaction(conn, sql)
    n_run_tables = len(cur.fetchall())

    pbar = tqdm(range(1), file=sys.stdout)
    pbar.set_description("Upgrading database; v8 -> v9")

    if n_run_tables == 1:
        _IX_runs_captured_run_id = """
                                CREATE INDEX
                                IF NOT EXISTS IX_runs_captured_run_id
                                ON runs (captured_run_id DESC)
                                """
        with atomic(conn) as connection:
            # iterate through the pbar for the sake of the side effect; it
            # prints that the database is being upgraded
            for _ in pbar:
                transaction(connection, _IX_runs_captured_run_id)
    else:
        raise RuntimeError(f"found {n_run_tables} runs tables expected 1")
