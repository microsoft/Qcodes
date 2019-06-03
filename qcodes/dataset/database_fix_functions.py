"""
Sometimes it happens that databases are put into inconsistent/corrupt states.
This module contains functions to remedy known issues.
"""
import json
import logging
from typing import Dict, Sequence

from tqdm import tqdm

import qcodes.dataset.descriptions.versioning.v0 as v0
import qcodes.dataset.descriptions.versioning.serialization as serial
from qcodes.dataset.sqlite.connection import ConnectionPlus, atomic, \
    atomic_transaction
from qcodes.dataset.sqlite.db_upgrades import get_user_version
from qcodes.dataset.sqlite.queries import get_parameters, \
    get_run_description, update_run_description, _update_run_description
from qcodes.dataset.sqlite.query_helpers import one, select_one_where


log = logging.getLogger(__name__)


def fix_version_4a_run_description_bug(conn: ConnectionPlus) -> Dict[str, int]:
    """
    Fix function to fix a bug where the RunDescriber accidentally wrote itself
    to string using the (new) InterDependencies_ object instead of the (old)
    InterDependencies object. After the first call, this function should be
    idempotent.


    Args:
        conn: the connection to the database

    Returns:
        A dict with the fix results ('runs_inspected', 'runs_fixed')
    """

    user_version = get_user_version(conn)

    if not user_version == 4:
        raise RuntimeError('Database of wrong version. Will not apply fix. '
                           'Expected version 4, found version {user_version}')

    no_of_runs_query = "SELECT max(run_id) FROM runs"
    no_of_runs = one(atomic_transaction(conn, no_of_runs_query), 'max(run_id)')
    no_of_runs = no_of_runs or 0

    with atomic(conn) as conn:

        pbar = tqdm(range(1, no_of_runs+1))
        pbar.set_description("Fixing database")

        # collect some metrics
        runs_inspected = 0
        runs_fixed = 0

        old_style_keys = ['paramspecs']
        new_style_keys = ['parameters', 'dependencies', 'inferences',
                          'standalones']

        for run_id in pbar:

            desc_str = get_run_description(conn, run_id)
            desc_ser = json.loads(desc_str)
            idps_ser = desc_ser['interdependencies']

            if list(idps_ser.keys()) == old_style_keys:
                pass
            elif list(idps_ser.keys()) == new_style_keys:
                # we cheat a little here; we introduce a RunDescriber version
                # number into a DB of schema version 4. That should be okay,
                # since the only rule is that DBs of schema versions > 4
                # MUST have a versioned RunDescriber,
                desc_ser['version'] = 1
                new_desc = serial.deserialize_to_current(desc_ser)
                old_desc_ser = serial.serialize_to_version(new_desc, 0)
                # the old_desc_ser has a "version" attribute with value 0,
                # so let's remove it.
                old_desc_ser.pop('version')
                json_str = json.dumps(old_desc_ser)
                _update_run_description(conn, run_id, json_str)
                runs_fixed += 1
            else:
                raise RuntimeError(f'Invalid runs_description for run_id: '
                                   f'{run_id}')

            runs_inspected += 1

    return {'runs_inspected': runs_inspected, 'runs_fixed': runs_fixed}


def fix_wrong_run_descriptions(conn: ConnectionPlus,
                               run_ids: Sequence[int]) -> None:
    """
    NB: This is a FIX function. Do not use it unless your database has been
    diagnosed with the problem that this function fixes.

    Overwrite faulty run_descriptions by using information from the layouts and
    dependencies tables. If a correct description is found for a run, that
    run is left untouched.

    Args:
        conn: The connection to the database
        run_ids: The runs to (potentially) fix
    """

    user_version = get_user_version(conn)

    if not user_version == 3:
        raise RuntimeError('Database of wrong version. Will not apply fix. '
                           'Expected version 3, found version {user_version}')


    log.info('[*] Fixing run descriptions...')
    for run_id in run_ids:
        trusted_paramspecs = get_parameters(conn, run_id)
        trusted_desc = v0.RunDescriber(
            v0.InterDependencies(*trusted_paramspecs))

        actual_desc_str = select_one_where(conn, "runs",
                                           "run_description",
                                           "run_id", run_id)

        trusted_json = serial.make_json_in_version(trusted_desc, 0)

        if actual_desc_str == trusted_json:
            log.info(f'[+] Run id: {run_id} had an OK description')
        else:
            log.info(f'[-] Run id: {run_id} had a broken description. '
                     f'Description found: {actual_desc_str}')
            update_run_description(conn, run_id, trusted_json)
            log.info(f'    Run id: {run_id} has been updated.')
