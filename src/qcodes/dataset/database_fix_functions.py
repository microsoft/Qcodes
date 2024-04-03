"""
Sometimes it happens that databases are put into inconsistent/corrupt states.
This module contains functions to remedy known issues.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

from tqdm import tqdm

import qcodes.dataset.descriptions.versioning.serialization as serial
from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.descriptions.versioning import v0
from qcodes.dataset.descriptions.versioning.converters import old_to_new
from qcodes.dataset.sqlite.connection import ConnectionPlus, atomic, atomic_transaction
from qcodes.dataset.sqlite.db_upgrades.version import get_user_version
from qcodes.dataset.sqlite.queries import (
    _get_parameters,
    _update_run_description,
    get_run_description,
    update_run_description,
)
from qcodes.dataset.sqlite.query_helpers import one, select_one_where

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qcodes.dataset.descriptions.versioning.rundescribertypes import (
        RunDescriberV1Dict,
    )

log = logging.getLogger(__name__)


def fix_version_4a_run_description_bug(conn: ConnectionPlus) -> dict[str, int]:
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
                old_desc_ser = \
                    _convert_run_describer_v1_like_dict_to_v0_like_dict(
                        desc_ser)
                json_str = json.dumps(old_desc_ser)
                _update_run_description(conn, run_id, json_str)
                runs_fixed += 1
            else:
                raise RuntimeError(f'Invalid runs_description for run_id: '
                                   f'{run_id}')

            runs_inspected += 1

    return {'runs_inspected': runs_inspected, 'runs_fixed': runs_fixed}


def _convert_run_describer_v1_like_dict_to_v0_like_dict(
    new_desc_dict: RunDescriberV1Dict,
) -> dict[str, Any]:
    """
    This function takes the given dict which is expected to be
    representation of `RunDescriber` with `InterDependencies_` (underscore!)
    object and without "version" field, and converts it to a dict that is a
    representation of the `RunDescriber` object with `InterDependencies`
    (no underscore!) object and without "version" field.
    """
    new_desc_dict = new_desc_dict.copy()
    # We intend to use conversion methods from `serialization` module,
    # but those work only with RunDescriber representations that have
    # "version" field. So first, the "version" field with correct value is
    # added.
    new_desc_dict['version'] = 1
    # Out of that dict we create RunDescriber object of the current version
    # (regardless of what the current version is).
    new_desc = serial.from_dict_to_current(new_desc_dict)
    # The RunDescriber of the current version gets converted to a dictionary
    # that represents a RunDescriber object of version 0 - this is the one
    # that has InterDependencies object in it (not the InterDependencies_ one).
    old_desc_dict = cast(dict[str, Any], serial.to_dict_as_version(new_desc, 0))
    # Lastly, the "version" field is removed.
    old_desc_dict.pop('version')
    return old_desc_dict


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
        trusted_paramspecs = _get_parameters(conn, run_id)
        interdeps = v0.InterDependencies(*trusted_paramspecs)
        interdeps_ = old_to_new(interdeps)
        trusted_desc = RunDescriber(interdeps_)

        actual_desc_str = select_one_where(conn, "runs",
                                           "run_description",
                                           "run_id", run_id)

        trusted_json = serial.to_json_as_version(trusted_desc, 0)

        if actual_desc_str == trusted_json:
            log.info(f'[+] Run id: {run_id} had an OK description')
        else:
            log.info(f'[-] Run id: {run_id} had a broken description. '
                     f'Description found: {actual_desc_str}')
            update_run_description(conn, run_id, trusted_json)
            log.info(f'    Run id: {run_id} has been updated.')
