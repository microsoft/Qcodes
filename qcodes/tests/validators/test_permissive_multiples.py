import numpy as np
import pytest
from qcodes.utils.validators import PermissiveMultiples

divisors = [40e-9, -1, 0.2225, 1 / 3, np.pi / 2]

multiples = [[800e-9, -40e-9, 0, 1],
             [3, -4, 0, -1, 1],
             [1.5575, -167.9875, 0],
             [2 / 3, 3, 1, 0, -5 / 3, 1e4],
             [np.pi, 5 * np.pi / 2, 0, -np.pi / 2]]

not_multiples = [[801e-9, 4.002e-5],
                 [1.5, 0.9999999],
                 [0.2226],
                 [0.6667, 28 / 9],
                 [3 * np.pi / 4]]


def test_passing():
    for divind, div in enumerate(divisors):
        val = PermissiveMultiples(div)
        for mult in multiples[divind]:
            val.validate(mult)


def test_not_passing():
    for divind, div in enumerate(divisors):
        val = PermissiveMultiples(div)
        for mult in not_multiples[divind]:
            with pytest.raises(ValueError):
                val.validate(mult)


# finally, a quick test that the precision is indeed setable
def test_precision():
    pm_lax = PermissiveMultiples(35e-9, precision=3e-9)
    pm_lax.validate(72e-9)
    pm_strict = PermissiveMultiples(35e-9, precision=1e-10)
    with pytest.raises(ValueError):
        pm_strict.validate(70.2e-9)


def test_valid_values():
    for div in divisors:
        pm = PermissiveMultiples(div)
        for val in pm.valid_values:
            pm.validate(val)
