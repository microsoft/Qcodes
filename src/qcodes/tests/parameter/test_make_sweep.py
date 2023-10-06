import numpy as np
import pytest

from qcodes.parameters.sweep_values import make_sweep


def test_good_calls() -> None:
    swp = make_sweep(1, 3, num=6)
    assert swp == [1, 1.4, 1.8, 2.2, 2.6, 3]

    swp = make_sweep(1, 3, step=0.5)
    assert swp == [1, 1.5, 2, 2.5, 3]

    # with step, test a lot of combinations with weird fractions
    # to make sure we don't fail on a rounding error
    for r in np.linspace(1, 4, 15):
        for steps in range(5, 55, 6):
            step = r / steps
            swp = make_sweep(1, 1 + r, step=step)
            assert len(swp) == steps + 1
            assert swp[0] == 1
            assert swp[-1] == 1 + r


def test_bad_calls() -> None:
    with pytest.raises(AttributeError):
        make_sweep(1, 3, num=3, step=1)

    with pytest.raises(ValueError):
        make_sweep(1, 3)

    # this first one should succeed
    make_sweep(1, 3, step=1)
    # but if we change step slightly (more than the tolerance of
    # 1e-10 steps) it will fail.
    with pytest.raises(ValueError):
        make_sweep(1, 3, step=1.00000001)
    with pytest.raises(ValueError):
        make_sweep(1, 3, step=0.99999999)
