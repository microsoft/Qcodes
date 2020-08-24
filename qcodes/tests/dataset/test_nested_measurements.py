import pytest

import numpy as np
from numpy.testing import assert_allclose

from qcodes.dataset.measurements import Measurement


@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize("bg_writing", [True, False])
def test_mixing_array_and_numeric(DAC, DMM, bg_writing):
    """
    Test that mixing array and numeric types is okay
    """
    meas1 = Measurement()
    meas1.register_parameter(DAC.ch1)
    meas1.register_parameter(DMM.v1, setpoints=(DAC.ch1,))

    meas2 = Measurement()
    meas2.register_parameter(DAC.ch2)
    meas2.register_parameter(DMM.v2, setpoints=(DAC.ch2,))

    with meas1.run(write_in_background=bg_writing) as ds1, meas2.run(write_in_background=bg_writing) as ds2:
        for i in range(10):
            DAC.ch1.set(i)
            DAC.ch2.set(i)
            ds1.add_result((DAC.ch1, i),
                           (DMM.v1, DMM.v1()))
            ds2.add_result((DAC.ch2, i),
                           (DMM.v2, DMM.v2()))

    data1 = ds1.dataset.get_parameter_data()["dummy_dmm_v1"]
    assert len(data1.keys()) == 2
    assert "dummy_dmm_v1" in data1.keys()
    assert "dummy_dac_ch1" in data1.keys()
    assert_allclose(data1["dummy_dmm_v1"], np.zeros(10))
    assert_allclose(data1["dummy_dac_ch1"], np.arange(10))

    data2 = ds2.dataset.get_parameter_data()["dummy_dmm_v2"]
    assert len(data2.keys()) == 2
    assert "dummy_dmm_v2" in data2.keys()
    assert "dummy_dac_ch2" in data2.keys()
    assert_allclose(data2["dummy_dmm_v2"], np.zeros(10))
    assert_allclose(data2["dummy_dac_ch2"], np.arange(10))
