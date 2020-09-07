"""
Test that multiple datasets can coexist as expected
"""
import pytest

from qcodes import new_experiment
from qcodes.dataset.data_set import DataSet


def test_foreground_after_background_raises(empty_temp_db_connection):
    new_experiment("test", "test1", conn=empty_temp_db_connection)
    ds1 = DataSet(conn=empty_temp_db_connection)
    ds1.mark_started(start_bg_writer=True)

    ds2 = DataSet(conn=empty_temp_db_connection)
    with pytest.raises(RuntimeError, match="All datasets written"):
        ds2.mark_started(start_bg_writer=False)


def test_background_after_foreground_raises(empty_temp_db_connection):
    new_experiment("test", "test1", conn=empty_temp_db_connection)
    ds1 = DataSet(conn=empty_temp_db_connection)
    ds1.mark_started(start_bg_writer=False)

    ds2 = DataSet(conn=empty_temp_db_connection)
    with pytest.raises(RuntimeError, match="All datasets written"):
        ds2.mark_started(start_bg_writer=True)


def test_background_twice(empty_temp_db_connection):
    new_experiment("test", "test1", conn=empty_temp_db_connection)
    ds1 = DataSet(conn=empty_temp_db_connection)
    ds1.mark_started(start_bg_writer=True)

    ds2 = DataSet(conn=empty_temp_db_connection)
    ds2.mark_started(start_bg_writer=True)


def test_foreground_twice(empty_temp_db_connection):
    new_experiment("test", "test1", conn=empty_temp_db_connection)
    ds1 = DataSet(conn=empty_temp_db_connection)
    ds1.mark_started(start_bg_writer=False)

    ds2 = DataSet(conn=empty_temp_db_connection)
    ds2.mark_started(start_bg_writer=False)


def test_foreground_after_background_non_concurrent(empty_temp_db_connection):
    new_experiment("test", "test1", conn=empty_temp_db_connection)
    ds1 = DataSet(conn=empty_temp_db_connection)
    ds1.mark_started(start_bg_writer=True)
    ds1.mark_completed()

    ds2 = DataSet(conn=empty_temp_db_connection)
    ds2.mark_started(start_bg_writer=False)
    ds2.mark_completed()

    ds3 = DataSet(conn=empty_temp_db_connection)
    ds3.mark_started(start_bg_writer=True)
    ds3.mark_completed()
