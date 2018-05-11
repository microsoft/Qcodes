import pytest
import qcodes
from qcodes.instrument_drivers.Lakeshore.Model_336 import Model_336
import qcodes.instrument.sims as sims

visalib = sims.__file__.replace('__init__.py', 'lakeshore_model336.yaml@sim')


def test_instantiation_model_336():
    ls = Model_336('lakeshore', 'GPIB::2::65535::INSTR', visalib=visalib, device_clear=False)