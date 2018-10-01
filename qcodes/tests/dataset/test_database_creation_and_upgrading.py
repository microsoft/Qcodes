from contextlib import contextmanager
from copy import deepcopy
import logging

import pytest

import qcodes as qc
from qcodes import new_experiment, new_data_set, ParamSpec
from qcodes.dataset.sqlite_base import connect
from qcodes.tests.dataset.temporary_databases import (empty_temp_db,
                                                      experiment)
from qcodes.dataset.sqlite_base import (update_GUIDs,
                                        get_user_version,
                                        atomic_transaction,
                                        perform_db_upgrade_0_to_1)

from qcodes.dataset.guids import parse_guid


@contextmanager
def location_and_station_set_to(location: int, work_station: int):
    cfg = qc.Config()
    old_cfg = deepcopy(cfg.current_config)
    cfg['GUID_components']['location'] = location
    cfg['GUID_components']['work_station'] = work_station
    cfg.save_to_home()

    yield

    cfg.current_config = old_cfg
    cfg.save_to_home()


@pytest.mark.usefixtures("empty_temp_db")
def test_tables_exist():
    for version in [-1, 0, 1]:
        conn = connect(qc.config["core"]["db_location"],
                       qc.config["core"]["db_debug"],
                       version=version)
        cursor = conn.execute("select sql from sqlite_master"
                              " where type = 'table'")
        expected_tables = ['experiments', 'runs', 'layouts', 'dependencies']
        rows = [row for row in cursor]
        assert len(rows) == len(expected_tables)
        for row, expected_table in zip(rows, expected_tables):
            assert expected_table in row['sql']
        conn.close()


def test_perform_actual_upgrade_0_to_1():
    # we cannot use the empty_temp_db, since that has already called connect
    # and is therefore latest version already
    connection = connect(':memory:', debug=False,
                         version=0)

    assert get_user_version(connection) == 0

    guid_table_query = "SELECT guid FROM runs"

    with pytest.raises(RuntimeError):
        atomic_transaction(connection, guid_table_query)

    perform_db_upgrade_0_to_1(connection)
    assert get_user_version(connection) == 1

    c = atomic_transaction(connection, guid_table_query)
    assert len(c.fetchall()) == 0


@pytest.mark.usefixtures("empty_temp_db")
def test_update_existing_guids(caplog):

    old_loc = 101
    old_ws = 1200

    new_loc = 232
    new_ws = 52123

    # prepare five runs with different location and work station codes

    with location_and_station_set_to(0, 0):
        new_experiment('test', sample_name='test_sample')

        ds1 = new_data_set('ds_one')
        xparam = ParamSpec('x', 'numeric')
        ds1.add_parameter(xparam)
        ds1.add_result({'x': 1})

        ds2 = new_data_set('ds_two')
        ds2.add_parameter(xparam)
        ds2.add_result({'x': 2})

        guid_comps_1 = parse_guid(ds1.guid)
        assert guid_comps_1['location'] == 0
        assert guid_comps_1['work_station'] == 0

        guid_comps_2 = parse_guid(ds2.guid)
        assert guid_comps_2['location'] == 0
        assert guid_comps_2['work_station'] == 0

    with location_and_station_set_to(0, old_ws):
        ds3 = new_data_set('ds_three')
        xparam = ParamSpec('x', 'numeric')
        ds3.add_parameter(xparam)
        ds3.add_result({'x': 3})

    with location_and_station_set_to(old_loc, 0):
        ds4 = new_data_set('ds_four')
        xparam = ParamSpec('x', 'numeric')
        ds4.add_parameter(xparam)
        ds4.add_result({'x': 4})

    with location_and_station_set_to(old_loc, old_ws):
        ds5 = new_data_set('ds_five')
        xparam = ParamSpec('x', 'numeric')
        ds5.add_parameter(xparam)
        ds5.add_result({'x': 5})

    with location_and_station_set_to(new_loc, new_ws):

        caplog.clear()
        expected_levels = ['INFO',
                           'INFO', 'INFO',
                           'INFO', 'INFO',
                           'INFO', 'WARNING',
                           'INFO', 'WARNING',
                           'INFO', 'INFO']

        with caplog.at_level(logging.INFO):
            update_GUIDs(ds1.conn)

            for record, lvl in zip(caplog.records, expected_levels):
                assert record.levelname == lvl

        guid_comps_1 = parse_guid(ds1.guid)
        assert guid_comps_1['location'] == new_loc
        assert guid_comps_1['work_station'] == new_ws

        guid_comps_2 = parse_guid(ds2.guid)
        assert guid_comps_2['location'] == new_loc
        assert guid_comps_2['work_station'] == new_ws

        guid_comps_3 = parse_guid(ds3.guid)
        assert guid_comps_3['location'] == 0
        assert guid_comps_3['work_station'] == old_ws

        guid_comps_4 = parse_guid(ds4.guid)
        assert guid_comps_4['location'] == old_loc
        assert guid_comps_4['work_station'] == 0

        guid_comps_5 = parse_guid(ds5.guid)
        assert guid_comps_5['location'] == old_loc
        assert guid_comps_5['work_station'] == old_ws
