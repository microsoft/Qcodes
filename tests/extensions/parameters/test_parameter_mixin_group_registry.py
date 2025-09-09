import pytest

from qcodes.extensions.parameters import GroupRegistryParameterMixin


class GroupRegistryTestParameterMixin(GroupRegistryParameterMixin):
    pass


def test_register_and_trigger_group_callbacks():
    calls = []
    GroupRegistryTestParameterMixin._group_registry.clear()

    def cb1():
        calls.append("cb1")

    def cb2():
        calls.append("cb2")

    GroupRegistryTestParameterMixin.register_group_callback("alpha", cb1)
    GroupRegistryTestParameterMixin.register_group_callback("alpha", cb2)
    GroupRegistryTestParameterMixin.trigger_group("alpha")

    assert calls == ["cb1", "cb2"]


def test_trigger_group_warns_if_none():
    GroupRegistryTestParameterMixin._group_registry.clear()
    with pytest.warns(UserWarning, match="No callbacks registered for group 'missing'"):
        GroupRegistryTestParameterMixin.trigger_group("missing")


def test_warning_no_callbacks_for_group() -> None:
    with pytest.warns(UserWarning, match="No callbacks registered for group"):
        GroupRegistryParameterMixin.trigger_group("empty_group")


def test_multiple_callbacks_in_group():
    """
    Test that multiple callbacks registered to a group are all called in order.
    """
    call_order = []
    GroupRegistryTestParameterMixin._group_registry.clear()

    def callback_one():
        call_order.append("callback_one")

    def callback_two():
        call_order.append("callback_two")

    GroupRegistryTestParameterMixin.register_group_callback(
        "multi_callback_group", callback_one
    )
    GroupRegistryTestParameterMixin.register_group_callback(
        "multi_callback_group", callback_two
    )

    GroupRegistryTestParameterMixin.trigger_group("multi_callback_group")
    assert call_order == ["callback_one", "callback_two"]


def test_callback_execution_order():
    """
    Test that callbacks are executed in the order they were registered.
    """
    execution_sequence = []
    GroupRegistryTestParameterMixin._group_registry.clear()

    def first_callback():
        execution_sequence.append("first")

    def second_callback():
        execution_sequence.append("second")

    GroupRegistryTestParameterMixin.register_group_callback(
        "order_group", first_callback
    )
    GroupRegistryTestParameterMixin.register_group_callback(
        "order_group", second_callback
    )

    GroupRegistryTestParameterMixin.trigger_group("order_group")
    assert execution_sequence == ["first", "second"]
