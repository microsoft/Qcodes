"""
This module encapsulates the initial schema of the QCoDeS database. The
schema is being adjusted by upgrade functions, if needed, see more on this
in :mod:`db_upgrades` module.
"""
from __future__ import annotations

from qcodes.dataset.sqlite.connection import ConnectionPlus, atomic, transaction


def init_db(conn: ConnectionPlus) -> None:
    with atomic(conn) as conn:
        transaction(conn, _experiment_table_schema)
        transaction(conn, _runs_table_schema)
        transaction(conn, _layout_table_schema)
        transaction(conn, _dependencies_table_schema)


_experiment_table_schema = """
CREATE  TABLE IF NOT EXISTS experiments (
    -- this will autoncrement by default if
    -- no value is specified on insert
    exp_id INTEGER PRIMARY KEY,
    name TEXT,
    sample_name TEXT,
    start_time INTEGER,
    end_time INTEGER,
    -- this is the last counter registered
    -- 1 based
    run_counter INTEGER,
    -- this is the formatter strin used to cosntruct
    -- the run name
    format_string TEXT
-- TODO: maybe I had a good reason for this doulbe primary key
--    PRIMARY KEY (exp_id, start_time, sample_name)
);
"""

_runs_table_schema = """
CREATE TABLE IF NOT EXISTS runs (
    -- this will autoincrement by default if
    -- no value is specified on insert
    run_id INTEGER PRIMARY KEY,
    exp_id INTEGER,
    -- friendly name for the run
    name TEXT,
    -- the name of the table which stores
    -- the actual results
    result_table_name TEXT,
    -- this is the run counter in its experiment 0 based
    result_counter INTEGER,
    ---
    run_timestamp INTEGER,
    completed_timestamp INTEGER,
    is_completed BOOL,
    parameters TEXT,
    -- metadata fields are added dynamically
    FOREIGN KEY(exp_id)
    REFERENCES
        experiments(exp_id)
);
"""

_layout_table_schema = """
CREATE TABLE IF NOT EXISTS layouts (
    layout_id INTEGER PRIMARY KEY,
    run_id INTEGER,
    -- name matching column name in result table
    parameter TEXT,
    label TEXT,
    unit TEXT,
    inferred_from TEXT,
    FOREIGN KEY(run_id)
    REFERENCES
        runs(run_id)
);
"""

_dependencies_table_schema = """
CREATE TABLE IF NOT EXISTS dependencies (
    dependent INTEGER,
    independent INTEGER,
    axis_num INTEGER
);
"""
