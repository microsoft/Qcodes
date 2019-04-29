import os

import pytest

from qcodes.dataset.database_fix_functions import fix_version_4a_run_description_bug
import qcodes.tests.dataset
from qcodes.tests.dataset.temporary_databases import temporarily_copied_DB

fixturepath = os.sep.join(qcodes.tests.dataset.__file__.split(os.sep)[:-1])
fixturepath = os.path.join(fixturepath, 'fixtures')


def test_version_4a_bugfix():
    v1fixpath = os.path.join(fixturepath, 'db_files', 'version4a')

    dbname_old = os.path.join(v1fixpath, 'some_runs.db')

    if not os.path.exists(dbname_old):
        pytest.skip("No db-file fixtures found. You can generate test db-files"
                    " using the scripts in the legacy_DB_generation folder")

    with temporarily_copied_DB(dbname_old, debug=False, version=4) as conn:

        dd = fix_version_4a_run_description_bug(conn)

        assert dd['runs_inspected'] == 10
        assert dd['runs_fixed'] == 10

        dd = fix_version_4a_run_description_bug(conn)

        assert dd['runs_inspected'] == 10
        assert dd['runs_fixed'] == 0
