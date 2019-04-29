"""
Sometimes it happens that databases are put into inconsistent/corrupt states.
This module contains functions to remedy known issues.
"""
import json
from typing import Dict

from tqdm import tqdm

from qcodes.dataset.sqlite_base import (ConnectionPlus, get_user_version,
                                        atomic)
from qcodes.dataset.descriptions import RunDescriber
from qcodes.dataset.sqlite_base import (get_run_description,
                                        update_run_description,
                                        one, atomic_transaction)


def fix_version_4a_run_description_bug(conn: ConnectionPlus) -> Dict[str, int]:
    """
    Fix function to fix a bug where the RunDescriber accidentally wrote itself
    to string using the (new) InterDependencies_ object instead of the (old)
    InterDependencies object. After the first run, this function should be
    idempotent.


    Args:
        conn: the connection to the database

    Returns:
        A dict with the fix results ('runs_inspected', 'runs_fixed')
    """

    if not get_user_version(conn) == 4:
        raise RuntimeError('Database of wrong version. Will not apply fix.')

    no_of_runs_query = "SELECT max(run_id) FROM runs"
    no_of_runs = one(atomic_transaction(conn, no_of_runs_query), 'max(run_id)')
    no_of_runs = no_of_runs or 0

    with atomic(conn) as conn:

        pbar = tqdm(range(1, no_of_runs+1))
        pbar.set_description("Fixing database")

        # collect some metrics
        runs_inspected = 0
        runs_fixed = 0

        for run_id in pbar:

            desc_str = get_run_description(conn, run_id)
            desc_ser = json.loads(desc_str)
            idps_ser = desc_ser['interdependencies']

            if RunDescriber._is_description_old_style(idps_ser):
                pass
            else:
                new_desc = RunDescriber.from_json(desc_str)
                update_run_description(conn, run_id, new_desc.to_json())
                runs_fixed += 1

            runs_inspected += 1

    return {'runs_inspected': runs_inspected, 'runs_fixed': runs_fixed}
