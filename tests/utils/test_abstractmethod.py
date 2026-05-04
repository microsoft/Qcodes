"""
Tests for qcodes.utils.abstractmethod - custom abstract method decorator.
"""

from qcodes.utils.abstractmethod import qcodes_abstractmethod


def test_decorator_sets_attribute() -> None:
    """Test that the decorator sets __qcodes_is_abstract_method__ to True."""

    @qcodes_abstractmethod
    def my_func() -> None:
        pass

    assert hasattr(my_func, "__qcodes_is_abstract_method__")
    assert my_func.__qcodes_is_abstract_method__ is True  # type: ignore[attr-defined]


def test_decorator_returns_same_function() -> None:
    """Test that the decorator returns the original function object."""

    def my_func() -> None:
        pass

    result = qcodes_abstractmethod(my_func)
    assert result is my_func


def test_decorated_function_is_still_callable() -> None:
    """Test that the decorated function can still be called."""

    @qcodes_abstractmethod
    def my_func(x: int) -> int:
        return x * 2

    assert my_func(5) == 10


def test_class_with_qcodes_abstractmethod_can_be_instantiated() -> None:
    """Test that unlike abc.abstractmethod, classes can still be instantiated."""

    class MyClass:
        @qcodes_abstractmethod
        def my_method(self) -> str:
            return "base"

    instance = MyClass()
    assert instance.my_method() == "base"


def test_undecorated_function_lacks_attribute() -> None:
    """Test that undecorated functions don't have the attribute."""

    def regular_func() -> None:
        pass

    assert not hasattr(regular_func, "__qcodes_is_abstract_method__")


def test_multiple_methods_decorated() -> None:
    """Test that multiple methods in a class can be decorated independently."""

    class MyClass:
        @qcodes_abstractmethod
        def method_a(self) -> None:
            pass

        @qcodes_abstractmethod
        def method_b(self) -> None:
            pass

        def method_c(self) -> None:
            pass

    assert hasattr(MyClass.method_a, "__qcodes_is_abstract_method__")
    assert hasattr(MyClass.method_b, "__qcodes_is_abstract_method__")
    assert not hasattr(MyClass.method_c, "__qcodes_is_abstract_method__")
