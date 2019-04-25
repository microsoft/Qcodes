import time
from copy import deepcopy
from contextlib import contextmanager

from hypothesis import given, settings
import hypothesis.strategies as hst
import numpy as np

from qcodes.dataset.guids import (generate_guid, parse_guid,
                                  set_guid_location_code,
                                  set_guid_work_station_code)
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
