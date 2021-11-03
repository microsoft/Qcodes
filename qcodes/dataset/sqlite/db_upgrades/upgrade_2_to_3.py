import json
import logging
import sys
from collections import defaultdict
from typing import DefaultDict, Dict, List, Mapping, Sequence, Tuple

from tqdm import tqdm

from qcodes.dataset.descriptions.param_spec import ParamSpec
from qcodes.dataset.descriptions.versioning.v0 import InterDependencies
from qcodes.dataset.sqlite.connection import (
    ConnectionPlus,
    atomic,
    atomic_transaction,
    transaction,
)
from qcodes.dataset.sqlite.query_helpers import one

log = logging.getLogger(__name__)


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


def _2to3_get_paramspecs(
    conn: ConnectionPlus,
    layout_ids: List[int],
    layouts: Mapping[int, Tuple[str, str, str, str]],
    dependencies: Mapping[int, Sequence[int]],
    deps: Sequence[int],
    indeps: Sequence[int],
    result_table_name: str,
) -> Dict[int, ParamSpec]:

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


def upgrade_2_to_3(conn: ConnectionPlus) -> None:
    """
    Perform the upgrade from version 2 to version 3

    Insert a new column, run_description, to the runs table and fill it out
    for exisitng runs with information retrieved from the layouts and
    dependencies tables represented as the json output of a RunDescriber
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

        pbar = tqdm(range(1, no_of_runs+1), file=sys.stdout)
        pbar.set_description("Upgrading database; v2 -> v3")

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
                desc_dict = {'interdependencies': interdeps._to_dict()}
                json_str = json.dumps(desc_dict)

            else:
                desc_dict = {'interdependencies':
                                 InterDependencies()._to_dict()}
                json_str = json.dumps(desc_dict)

            sql = f"""
                   UPDATE runs
                   SET run_description = ?
                   WHERE run_id == ?
                   """
            cur = conn.cursor()
            cur.execute(sql, (json_str, run_id))
            log.debug(f"Upgrade in transition, run number {run_id}: OK")
