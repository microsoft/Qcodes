import logging

import pytest

from .conftest import MemoryParameter


def test_step_ramp(caplog):
    p = MemoryParameter(name='test_step')
    p(42)
    assert p.set_values == [42]
    p.step = 1

    assert p.get_ramp_values(44.5, 1) == [43, 44, 44.5]

    p(44.5)
    assert p.set_values == [42, 43, 44, 44.5]

    # Assert that stepping does not impact ``cache.set`` call, and that
    # the value that is passed to ``cache.set`` call does not get
    # propagated to parameter's ``set_cmd``
    p.cache.set(40)
    assert p.get_latest() == 40
    assert p.set_values == [42, 43, 44, 44.5]

    # Test error conditions
    with caplog.at_level(logging.WARNING):
        assert p.get_ramp_values("A", 1) == []
        assert len(caplog.records) == 1
    with pytest.raises(RuntimeError):
        p.get_ramp_values((1, 2, 3), 1)
