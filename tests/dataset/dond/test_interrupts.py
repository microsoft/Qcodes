import pytest
from qcodes.dataset.dond.do_nd_utils import catch_interrupts, BreakConditionInterrupt

def test_keyboard_interrupt_handling():
    with pytest.raises(KeyboardInterrupt):
        with catch_interrupts() as get_interrupt_exception:
            raise KeyboardInterrupt
        assert isinstance(get_interrupt_exception(), KeyboardInterrupt)

def test_break_condition_interrupt_handling():
    with pytest.raises(BreakConditionInterrupt):
        with catch_interrupts() as get_interrupt_exception:
            raise BreakConditionInterrupt
        assert isinstance(get_interrupt_exception(), BreakConditionInterrupt)

def test_no_interrupt_handling():
    with catch_interrupts() as get_interrupt_exception:
        pass
    assert get_interrupt_exception() is None
