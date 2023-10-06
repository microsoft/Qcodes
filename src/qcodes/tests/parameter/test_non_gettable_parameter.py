from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.testing import assert_array_almost_equal

if TYPE_CHECKING:
    from pytest import LogCaptureFixture

from qcodes.parameters import Parameter


def test_setting_non_gettable_parameter_with_finite_step(
    caplog: LogCaptureFixture,
) -> None:
    initial_value = 0
    step_size = 0.1
    set_value = 1.0

    # when the parameter is initially set from
    # the initial_value the starting point is unknown
    # so this should cause a warning but the parameter should still be set
    with caplog.at_level(logging.WARNING):
        x = Parameter('x', initial_value=initial_value,
                      step=step_size,
                      set_cmd=None)
        assert len(caplog.records) == 1
        assert f"cannot sweep x from None to {initial_value}" in str(caplog.records[0])
    assert x.cache.get() == 0

    # afterwards the stepping should work as expected.
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        assert_array_almost_equal(
            np.array(x.get_ramp_values(set_value, step_size)),
            (np.arange(initial_value + step_size, set_value + step_size, step_size)),
        )
        x.set(set_value)
        assert x.cache.get() == set_value
        assert len(caplog.records) == 0
