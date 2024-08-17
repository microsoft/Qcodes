from __future__ import annotations

import json
import logging
import sys

from tqdm import tqdm

from qcodes.dataset.descriptions.versioning.v0 import InterDependencies
from qcodes.dataset.sqlite.connection import ConnectionPlus, atomic, atomic_transaction
from qcodes.dataset.sqlite.db_upgrades.upgrade_2_to_3 import (
    _2to3_get_dependencies,
    _2to3_get_deps,
    _2to3_get_indeps,
    _2to3_get_layout_ids,
    _2to3_get_layouts,
    _2to3_get_paramspecs,
    _2to3_get_result_tables,
)
from qcodes.dataset.sqlite.query_helpers import one

log = logging.getLogger(__name__)


def upgrade_3_to_4(conn: ConnectionPlus, show_progress_bar: bool = True) -> None:
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
    no_of_runs = one(atomic_transaction(conn, no_of_runs_query), "max(run_id)")
    no_of_runs = no_of_runs or 0

    # If one run fails, we want the whole upgrade to roll back, hence the
    # entire upgrade is one atomic transaction

    with atomic(conn) as atomic_conn:
        result_tables = _2to3_get_result_tables(atomic_conn)
        layout_ids_all = _2to3_get_layout_ids(atomic_conn)
        indeps_all = _2to3_get_indeps(atomic_conn)
        deps_all = _2to3_get_deps(atomic_conn)
        layouts = _2to3_get_layouts(atomic_conn)
        dependencies = _2to3_get_dependencies(atomic_conn)

        pbar = tqdm(
            range(1, no_of_runs + 1), file=sys.stdout, disable=not show_progress_bar
        )
        pbar.set_description("Upgrading database; v3 -> v4")

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

                paramspecs = _2to3_get_paramspecs(
                    atomic_conn,
                    layout_ids,
                    layouts,
                    dependencies,
                    dependents,
                    independents,
                    result_table_name,
                )

                interdeps = InterDependencies(*paramspecs.values())
                desc_dict = {"interdependencies": interdeps._to_dict()}
                json_str = json.dumps(desc_dict)

            else:
                desc_dict = {"interdependencies": InterDependencies()._to_dict()}
                json_str = json.dumps(desc_dict)

            sql = """
                   UPDATE runs
                   SET run_description = ?
                   WHERE run_id == ?
                   """
            cur = atomic_conn.cursor()
            cur.execute(sql, (json_str, run_id))
            log.debug(f"Upgrade in transition, run number {run_id}: OK")
