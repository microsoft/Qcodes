from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, cast

import numpy as np
import pytest

from qcodes.dataset.experiment_container import (
    load_or_create_experiment,
    new_experiment,
)
from qcodes.dataset.guid_helpers import guids_from_dir, guids_from_list_str
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.sqlite.connection import ConnectionPlus
from qcodes.dataset.sqlite.database import initialised_database_at
from qcodes.dataset.sqlite.queries import get_guids_from_multiple_run_ids
from qcodes.instrument.parameter import Parameter


def test_guids_from_dir(tmp_path: Path) -> None:
    def generate_local_run(dbpath: Path) -> str:
        with initialised_database_at(str(dbpath)):
            new_experiment(sample_name="fivehundredtest_sample",
                           name="fivehundredtest_name")

            p1 = Parameter('Voltage', set_cmd=None)
            p2 = Parameter('Current', get_cmd=np.random.randn)

            meas = Measurement()
            meas.register_parameter(p1).register_parameter(p2, setpoints=[p1])

            with meas.run() as datasaver:
                for v in np.linspace(0, 2, 250):
                    p1(v)
                    datasaver.add_result((p1, cast(float, p1())),
                                         (p2, cast(float, p2())))
            guid = datasaver.dataset.guid
            datasaver.flush_data_to_database(block=True)
        return guid

    paths_counts = [
        (tmp_path / 'subdir1' / 'dbfile1.db', 2),
        (tmp_path / 'subdir1' / 'dbfile2.db', 4),
        (tmp_path / 'subdir2' / 'dbfile1.db', 1),
        (tmp_path / 'dbfile1.db', 3),
    ]
    guids = defaultdict(list)
    for path, count in paths_counts:
        path.parent.mkdir(exist_ok=True, parents=True)
        for _ in range(count):
            guids[path].append(generate_local_run(path))
    dbdict, _ = guids_from_dir(tmp_path)
    assert dbdict == guids


def test_guids_from_list_str() -> None:
    guids = ['07fd7195-c51e-44d6-a085-fa8274cf00d6',
             '070d7195-c51e-44d6-a085-fa8274cf00d6']
    assert guids_from_list_str('') == tuple()
    assert guids_from_list_str("''") == tuple()
    assert guids_from_list_str('""') == tuple()
    assert guids_from_list_str(str(tuple())) == tuple()
    assert guids_from_list_str(str(list())) == tuple()
    assert guids_from_list_str(str({})) is None
    assert guids_from_list_str(str(guids)) == tuple(guids)
    assert guids_from_list_str(str([guids[0]])) == (guids[0],)
    assert guids_from_list_str(str(guids[0])) == (guids[0],)
    assert guids_from_list_str(str(tuple(guids))) == tuple(guids)
    extracted_guids = guids_from_list_str(str(set(guids)))
    assert extracted_guids is not None
    assert sorted(extracted_guids) == sorted(tuple(guids))


def test_many_guids_from_list_str() -> None:
    guids = [
        'aaaaaaaa-0d00-000d-0000-017662aded3d',
        'aaaaaaaa-0d00-000d-0000-017662ae5fec',
        'aaaaaaaa-0d00-000d-0000-017662b01bb7',
        'aaaaaaaa-0d00-000d-0000-017662b18452',
        'aaaaaaaa-0d00-000d-0000-017662b298c2',
        'aaaaaaaa-0d00-000d-0000-017662b2a878',
        'aaaaaaaa-0d00-000d-0000-01766827cfaf']
    assert guids_from_list_str(str(guids)) == tuple(guids)


def test_get_guids_from_multiple_run_ids(tmp_path: Path) -> None:
    def generate_local_exp(dbpath: Path) -> Tuple[List[str], ConnectionPlus]:
        with initialised_database_at(str(dbpath)):
            guids = []
            exp = load_or_create_experiment(experiment_name="test_guid")
            conn = exp.conn

            p1 = Parameter('Voltage', set_cmd=None)
            p2 = Parameter('Current', get_cmd=np.random.randn)

            meas = Measurement(exp=exp)
            meas.register_parameter(p1).register_parameter(p2, setpoints=[p1])

            # Meaure for 2 times to get 2 run ids and 2 guids
            for run in range(2):
                with meas.run() as datasaver:
                    for v in np.linspace(0*run, 2*run, 50):
                        p1(v)
                        datasaver.add_result((p1, cast(float, p1())),
                                             (p2, cast(float, p2())))
                guid = datasaver.dataset.guid
                guids.append(guid)
        return guids, conn

    path = tmp_path/'dbfile2.db'
    guids, conn = generate_local_exp(path)

    assert get_guids_from_multiple_run_ids(conn=conn, run_ids=[1, 2]) == guids

    assert len(guids) == 2

    with pytest.raises(RuntimeError, match="run id 3 does not"
                       " exist in the database"):
        get_guids_from_multiple_run_ids(conn=conn, run_ids=[1, 2, 3])
