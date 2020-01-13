import time
from copy import deepcopy
from contextlib import contextmanager
from uuid import uuid4

import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as hst
import numpy as np

from qcodes.dataset.guids import (generate_guid, parse_guid,
                                  set_guid_location_code,
                                  set_guid_work_station_code,
                                  validate_guid_format,
                                  filter_guids_by_parts)
from qcodes.config import Config, DotDict

@contextmanager
def protected_config():
    """
    Context manager to be used in all tests that modify the config to ensure
    that the config is left untouched even if the tests fail
    """
    ocfg: DotDict = Config().current_config
    original_config = deepcopy(ocfg)

    try:
        yield
    finally:
        cfg = Config()
        cfg.current_config = original_config
        cfg.save_to_home()


@settings(max_examples=50, deadline=1000)
@given(loc=hst.integers(0, 255), stat=hst.integers(0, 65535),
       smpl=hst.integers(0, 4294967295))
def test_generate_guid(loc, stat, smpl):
    # update config to generate a particular guid. Read it back to verify
    with protected_config():
        cfg = Config()
        cfg['GUID_components']['location'] = loc
        cfg['GUID_components']['work_station'] = stat
        cfg['GUID_components']['sample'] = smpl
        cfg.save_to_home()

        guid = generate_guid()
        gen_time = int(np.round(time.time()*1000))

        comps = parse_guid(guid)

        if smpl == 0:
            smpl = int('a'*8, base=16)

        assert comps['location'] == loc
        assert comps['work_station'] == stat
        assert comps['sample'] == smpl
        assert comps['time'] - gen_time < 2


@settings(max_examples=50, deadline=None)
@given(loc=hst.integers(-10, 350))
def test_set_guid_location_code(loc, monkeypatch):
    monkeypatch.setattr('builtins.input', lambda x: str(loc))

    orig_cfg = Config().current_config

    original_loc = orig_cfg['GUID_components']['location']

    with protected_config():
        set_guid_location_code()

        cfg = Config().current_config

        if 257 > loc > 0:
            assert cfg['GUID_components']['location'] == loc
        else:
            assert cfg['GUID_components']['location'] == original_loc


@settings(max_examples=50, deadline=1000)
@given(ws=hst.integers(-10, 17000000))
def test_set_guid_workstatio_code(ws, monkeypatch):
    monkeypatch.setattr('builtins.input', lambda x: str(ws))

    orig_cfg = Config().current_config

    original_ws = orig_cfg['GUID_components']['work_station']

    with protected_config():
        set_guid_work_station_code()

        cfg = Config().current_config

        if 16777216 > ws > 0:
            assert cfg['GUID_components']['work_station'] == ws
        else:
            assert cfg['GUID_components']['work_station'] == original_ws


@settings(max_examples=50, deadline=1000)
@given(locs=hst.lists(hst.integers(0, 255), min_size=2, max_size=2, unique=True),
       stats=hst.lists(hst.integers(0, 65535), min_size=2, max_size=2, unique=True),
       smpls=hst.lists(hst.integers(0, 4294967295), min_size=2, max_size=2, unique=True),
       )
def test_filter_guid(locs, stats, smpls):

    def make_test_guid(cfg, loc: int, smpl: int, stat: int):
        cfg['GUID_components']['location'] = loc
        cfg['GUID_components']['work_station'] = stat
        cfg['GUID_components']['sample'] = smpl
        cfg.save_to_home()

        guid = generate_guid()
        gen_time = int(np.round(time.time() * 1000))

        comps = parse_guid(guid)

        assert comps['location'] == loc
        assert comps['work_station'] == stat
        assert comps['sample'] == smpl
        assert comps['time'] - gen_time < 2

        return guid

    with protected_config():

        guids = []
        cfg = Config()

        corrected_smpls = [smpl if smpl != 0 else int('a' * 8, base=16)
                           for smpl in smpls]
        # there is a possibility that we could generate 0 and 2863311530, which
        # are considered equivalent since int('a' * 8, base=16) == 2863311530.
        # We want unique samples, so we exclude this case.
        assume(corrected_smpls[0] != corrected_smpls[1])

        # first we generate a guid that we are going to match against
        guids.append(make_test_guid(cfg, locs[0], corrected_smpls[0], stats[0]))

        # now generate some guids that will not match because one of the
        # components changed
        guids.append(make_test_guid(cfg, locs[1], corrected_smpls[0], stats[0]))
        guids.append(make_test_guid(cfg, locs[0], corrected_smpls[1], stats[0]))
        guids.append(make_test_guid(cfg, locs[0], corrected_smpls[0], stats[1]))

        assert len(guids) == 4

        # first filter on all parts. This should give exactly one matching guid
        filtered_guids = filter_guids_by_parts(guids,
                                               location=locs[0],
                                               sample_id=corrected_smpls[0],
                                               work_station=stats[0]
                                               )

        assert len(filtered_guids) == 1
        assert filtered_guids[0] == guids[0]

        # now filter on 2 components
        filtered_guids = filter_guids_by_parts(guids,
                                               location=locs[0],
                                               sample_id=corrected_smpls[0])
        assert len(filtered_guids) == 2
        assert filtered_guids[0] == guids[0]
        assert filtered_guids[1] == guids[3]

        filtered_guids = filter_guids_by_parts(guids,
                                               location=locs[0],
                                               work_station=stats[0])
        assert len(filtered_guids) == 2
        assert filtered_guids[0] == guids[0]
        assert filtered_guids[1] == guids[2]

        filtered_guids = filter_guids_by_parts(guids,
                                               sample_id=corrected_smpls[0],
                                               work_station=stats[0]
                                               )
        assert len(filtered_guids) == 2
        assert filtered_guids[0] == guids[0]
        assert filtered_guids[1] == guids[1]

        # now filter on 1 component
        filtered_guids = filter_guids_by_parts(guids,
                                               location=locs[0])
        assert len(filtered_guids) == 3
        assert filtered_guids[0] == guids[0]
        assert filtered_guids[1] == guids[2]
        assert filtered_guids[2] == guids[3]

        filtered_guids = filter_guids_by_parts(guids,
                                               work_station=stats[0])
        assert len(filtered_guids) == 3
        assert filtered_guids[0] == guids[0]
        assert filtered_guids[1] == guids[1]
        assert filtered_guids[2] == guids[2]

        filtered_guids = filter_guids_by_parts(guids,
                                               sample_id=corrected_smpls[0],
                                               )
        assert len(filtered_guids) == 3
        assert filtered_guids[0] == guids[0]
        assert filtered_guids[1] == guids[1]
        assert filtered_guids[2] == guids[3]


def test_validation():
    valid_guid = str(uuid4())
    validate_guid_format(valid_guid)

    with pytest.raises(ValueError):
        validate_guid_format(valid_guid[1:])
