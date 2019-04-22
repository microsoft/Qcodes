import os

import pytest

import qcodes
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.dependencies import InterDependencies_
from qcodes.dataset.descriptions import RunDescriber
from qcodes.dataset.sqlite_base import _fix_wrong_run_descriptions, \
    get_user_version
from qcodes.tests.dataset.temporary_databases import temporarily_copied_DB

fixturepath = os.sep.join(qcodes.tests.dataset.__file__.split(os.sep)[:-1])
fixturepath = os.path.join(fixturepath, 'fixtures')


def test_fix_wrong_run_descriptions():
    v3fixpath = os.path.join(fixturepath, 'db_files', 'version3')

    dbname_old = os.path.join(v3fixpath, 'some_runs_without_run_description.db')

    if not os.path.exists(dbname_old):
        pytest.skip(
            "No db-file fixtures found. You can generate test db-files"
            " using the scripts in the legacy_DB_generation folder")

    with temporarily_copied_DB(dbname_old, debug=False, version=3) as conn:

        assert get_user_version(conn) == 3

        ds1 = DataSet(conn=conn, run_id=1)
        expected_description = ds1.description

        empty_description = RunDescriber(InterDependencies_())

        _fix_wrong_run_descriptions(conn, [1, 2, 3, 4])

        ds2 = DataSet(conn=conn, run_id=2)
        assert expected_description == ds2.description

        ds3 = DataSet(conn=conn, run_id=3)
        assert expected_description == ds3.description

        ds4 = DataSet(conn=conn, run_id=4)
        assert empty_description == ds4.description
