import json
import os

import pytest

import qcodes.dataset.descriptions.versioning.serialization as serial
import tests.dataset
from qcodes.dataset.database_fix_functions import (
    fix_version_4a_run_description_bug,
    fix_wrong_run_descriptions,
)
from qcodes.dataset.descriptions.param_spec import ParamSpec
from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.descriptions.versioning import v0
from qcodes.dataset.descriptions.versioning.converters import old_to_new
from qcodes.dataset.sqlite.db_upgrades.version import get_user_version
from qcodes.dataset.sqlite.queries import get_run_description
from tests.common import skip_if_no_fixtures
from tests.dataset.conftest import temporarily_copied_DB

fixturepath = os.sep.join(tests.dataset.__file__.split(os.sep)[:-1])
fixturepath = os.path.join(fixturepath, "fixtures")


def test_version_4a_bugfix() -> None:
    v4fixpath = os.path.join(fixturepath, "db_files", "version4a")

    dbname_old = os.path.join(v4fixpath, "some_runs.db")

    skip_if_no_fixtures(dbname_old)

    with temporarily_copied_DB(dbname_old, debug=False, version=4) as conn:
        dd = fix_version_4a_run_description_bug(conn)

        assert dd["runs_inspected"] == 10
        assert dd["runs_fixed"] == 10

        # Ensure the structure of the run_description JSON after applying
        # the fix function
        for run_id in range(1, 10 + 1):
            rd_str = get_run_description(conn, run_id)
            rd_dict = json.loads(rd_str)
            assert list(rd_dict.keys()) == ["interdependencies"]
            assert list(rd_dict["interdependencies"].keys()) == ["paramspecs"]

        dd = fix_version_4a_run_description_bug(conn)

        assert dd["runs_inspected"] == 10
        assert dd["runs_fixed"] == 0


def test_version_4a_bugfix_raises() -> None:
    v3fixpath = os.path.join(fixturepath, "db_files", "version3")
    dbname_old = os.path.join(v3fixpath, "some_runs_without_run_description.db")

    skip_if_no_fixtures(dbname_old)

    with temporarily_copied_DB(dbname_old, debug=False, version=3) as conn:
        with pytest.raises(RuntimeError):
            fix_version_4a_run_description_bug(conn)


def test_fix_wrong_run_descriptions() -> None:
    v3fixpath = os.path.join(fixturepath, "db_files", "version3")

    dbname_old = os.path.join(v3fixpath, "some_runs_without_run_description.db")

    skip_if_no_fixtures(dbname_old)

    def make_ps(n):
        ps = ParamSpec(
            f"p{n}", label=f"Parameter {n}", unit=f"unit {n}", paramtype="numeric"
        )
        return ps

    paramspecs = [make_ps(n) for n in range(6)]
    paramspecs[2]._inferred_from = ["p0"]
    paramspecs[3]._inferred_from = ["p1", "p0"]
    paramspecs[4]._depends_on = ["p2", "p3"]
    paramspecs[5]._inferred_from = ["p0"]

    with temporarily_copied_DB(dbname_old, debug=False, version=3) as conn:
        assert get_user_version(conn) == 3

        expected_description = RunDescriber(
            old_to_new(v0.InterDependencies(*paramspecs))
        )

        empty_description = RunDescriber(old_to_new(v0.InterDependencies()))

        fix_wrong_run_descriptions(conn, [1, 2, 3, 4])

        for run_id in [1, 2, 3]:
            desc_str = get_run_description(conn, run_id)
            desc = serial.from_json_to_current(desc_str)
            assert desc == expected_description

        desc_str = get_run_description(conn, run_id=4)
        desc = serial.from_json_to_current(desc_str)
        assert desc == empty_description


def test_fix_wrong_run_descriptions_raises() -> None:
    v4fixpath = os.path.join(fixturepath, "db_files", "version4a")

    dbname_old = os.path.join(v4fixpath, "some_runs.db")

    skip_if_no_fixtures(dbname_old)

    with temporarily_copied_DB(dbname_old, debug=False, version=4) as conn:
        with pytest.raises(RuntimeError):
            fix_wrong_run_descriptions(conn, [1])
