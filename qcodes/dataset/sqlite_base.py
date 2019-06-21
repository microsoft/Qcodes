"""
Code of this module has been moved to `.sqlite.*`, mostly to
`.sqlite.queries`. This module now only re-imports the functions which it
used to contain, for backwards compatibility. Do not import functions from
this module because it will be removed soon.
"""
import warnings

from qcodes.dataset.sqlite.connection import ConnectionPlus, atomic, \
    transaction, atomic_transaction
from qcodes.dataset.sqlite.connection import make_connection_plus_from
from qcodes.dataset.sqlite.database import connect, \
    get_db_version_and_newest_available_version
from qcodes.dataset.sqlite.db_upgrades import _latest_available_version, \
    perform_db_upgrade, upgrader
from qcodes.dataset.sqlite.db_upgrades.version import get_user_version
from qcodes.dataset.sqlite.db_upgrades.version import set_user_version
from qcodes.dataset.sqlite.initial_schema import init_db
from qcodes.dataset.sqlite.queries import is_run_id_in_database, \
    _build_data_query, get_data, get_parameter_data, get_values, \
    get_parameter_tree_values, get_setpoints, \
    get_runid_from_expid_and_counter, \
    get_runid_from_guid, get_layout, get_layout_id, get_dependents, \
    get_dependencies, get_non_dependencies, get_parameter_dependencies, \
    new_experiment, mark_run_complete, completed, \
    get_completed_timestamp_from_run_id, get_guid_from_run_id, \
    finish_experiment, get_run_counter, get_experiments, \
    get_matching_exp_ids, \
    get_exp_ids_from_run_ids, get_last_experiment, get_runs, get_last_run, \
    run_exists, data_sets, format_table_name, _insert_run, \
    _update_experiment_run_counter, get_parameters, get_paramspec, \
    update_run_description, set_run_timestamp, add_parameter, \
    _add_parameters_to_layout_and_deps, _validate_table_name, \
    _create_run_table, create_run, get_run_description, get_metadata, \
    get_metadata_from_run_id, insert_meta_data, update_meta_data, \
    add_meta_data, get_experiment_name_from_experiment_id, \
    get_sample_name_from_experiment_id, get_run_timestamp_from_run_id, \
    update_GUIDs, remove_trigger, _unicode_categories, RUNS_TABLE_COLUMNS
from qcodes.dataset.sqlite.settings import SQLiteSettings
from qcodes.dataset.sqlite.query_helpers import many_many, one, many, \
    select_one_where, select_many_where, update_where, insert_values, \
    VALUES, insert_column
from qcodes.dataset.sqlite.query_helpers import insert_many_values, VALUE, \
    modify_values, modify_many_values, length, is_column_in_table, \
    sql_placeholder_string
from qcodes.dataset.sqlite.db_upgrades import perform_db_upgrade_0_to_1, \
    perform_db_upgrade_1_to_2, perform_db_upgrade_2_to_3, \
    perform_db_upgrade_3_to_4, perform_db_upgrade_4_to_5, \
    perform_db_upgrade_5_to_6

warnings.warn('The module `qcodes.dataset.sqlite_base` is deprecated.\n'
              'Public features are available at the import of `qcodes`.\n'
              'Private features are available in `qcodes.dataset.sqlite.*` '
              'modules.')
