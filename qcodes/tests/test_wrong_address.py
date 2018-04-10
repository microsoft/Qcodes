import pytest

import qcodes.instrument.sims as sims
from qcodes.instrument_drivers.Keysight.Keysight_34465A import Keysight_34465A
visalib = sims.__file__.replace('__init__.py', 'Keysight_34465A.yaml@sim')


def test_wrong_address():
    try:
        # wrong address
        keysight_sim = Keysight_34465A('keysight_34465A_sim',
                                       address='GPIB::1::6553512341234::INSTR',
                                       visalib=visalib)
    except:
        pass
    # right address
    keysight_sim = Keysight_34465A('keysight_34465A_sim',
                                   address='GPIB::1::65535::INSTR',
                                   visalib=visalib)
