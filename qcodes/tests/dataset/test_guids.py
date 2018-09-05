import time
from copy import deepcopy

from hypothesis import given, settings
import hypothesis.strategies as hst
import numpy as np

from qcodes.dataset.guids import generate_guid, parse_guid
from qcodes.config import Config, DotDict

ocfg: DotDict = Config().current_config
original_config = deepcopy(ocfg)

@settings(max_examples=50)
@given(loc=hst.integers(0, 255), stat=hst.integers(0, 65535),
       smpl=hst.integers(0, 4294967295))
def test_generate_guid(loc, stat, smpl):
    # update config to generate a particular guid. Read it back to verify
    try:
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
    finally:
        # important to leave this vital info untouched!
        cfg = Config()
        cfg.current_config = original_config
        cfg.save_to_home()
