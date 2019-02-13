import logging
import re

import pytest

import qcodes.instrument.sims as sims
from qcodes.instrument_drivers.Keysight.Keysight_34465A_submodules import \
    Keysight_34465A
from qcodes.tests.common import error_caused_by


visalib = sims.__file__.replace('__init__.py', 'Keysight_34465A.yaml@sim')

logging.basicConfig(level=logging.DEBUG)


def test_wrong_address():
    wrong_address = 'GPIB::2::INSTR'

    match_str = re.escape(f'ERROR: resource {wrong_address} not found')
    with pytest.raises(Exception, match=match_str) as exc_info:
        _ = Keysight_34465A('keysight_34465A_sim',
                            address=wrong_address,
                            visalib=visalib)

    assert error_caused_by(exc_info, wrong_address)

    right_address = 'GPIB::1::INSTR'  # from the simulation yaml file

    _ = Keysight_34465A('keysight_34465A_sim',
                        address=right_address,
                        visalib=visalib)
