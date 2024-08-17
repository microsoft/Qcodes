import warnings

import pytest

from qcodes.utils.deprecate import (
    QCoDeSDeprecationWarning,
    _catch_deprecation_warnings,
    assert_deprecated,
    assert_not_deprecated,
    deprecate,
    issue_deprecation_warning,
)


def test_assert_deprecated_raises() -> None:
    with assert_deprecated(
        "The use of this function is deprecated, because "
        'of this being a test. Use "a real function" as an '
        "alternative."
    ):
        issue_deprecation_warning(
            "use of this function", "of this being a test", "a real function"
        )


def test_assert_deprecated_does_not_raise_wrong_msg() -> None:
    with pytest.raises(AssertionError):
        with assert_deprecated("entirely different message"):
            issue_deprecation_warning("warning")


def test_assert_deprecated_does_not_raise_wrong_type() -> None:
    with pytest.raises(AssertionError):
        with assert_deprecated("entirely different message"):
            warnings.warn("warning")


def test_assert_not_deprecated_raises() -> None:
    with pytest.raises(AssertionError):
        with assert_not_deprecated():
            issue_deprecation_warning("something")


def test_assert_not_deprecated_does_not_raise() -> None:
    with assert_not_deprecated():
        warnings.warn("some other warning")


@pytest.mark.filterwarnings('ignore:The function "add_one" is deprecated,')
def test_similar_output() -> None:
    def _add_one(x):
        return 1 + x

    @deprecate(reason="this function is for private use only")
    def add_one(x):
        return _add_one(x)

    with assert_deprecated(
        "The function <add_one> is deprecated, because "
        "this function is for private use only."
    ):
        assert add_one(1) == _add_one(1)


@pytest.mark.filterwarnings("ignore:Some other warning")
def test_deprecated_context_manager() -> None:
    with _catch_deprecation_warnings() as ws:
        issue_deprecation_warning("something")
        issue_deprecation_warning("something more")
        warnings.warn("Some other warning")
    assert len(ws) == 2

    with pytest.warns(expected_warning=QCoDeSDeprecationWarning) as ws_2:
        issue_deprecation_warning("something")
        warnings.warn("Some other warning")
    assert len(ws_2) == 2


@deprecate(reason="this is a test")
class C:
    def __init__(self, attr: str):
        self.a = attr

    def method(self) -> None:
        self.a = "last called by method"

    @staticmethod
    def static_method(a: int) -> int:
        return a + 1

    @classmethod
    def class_method(cls, a: int) -> int:
        return a + 1

    @property
    def prop(self) -> str:
        return self.a

    @prop.setter
    def prop(self, val: str) -> None:
        self.a = val + "_prop"


def test_init_uninhibited() -> None:
    with pytest.warns(expected_warning=QCoDeSDeprecationWarning):
        c = C("pristine")
    assert c.a == "pristine"


def test_init_raises() -> None:
    with assert_deprecated("The class <C> is deprecated, because this is a test."):
        C("pristine")


def test_method_uninhibited() -> None:
    with pytest.warns(expected_warning=QCoDeSDeprecationWarning):
        c = C("pristine")
        c.method()
    assert c.a == "last called by method"


def test_method_raises() -> None:
    with pytest.warns(expected_warning=QCoDeSDeprecationWarning):
        c = C("pristine")

    with assert_deprecated(
        "The function <method> is deprecated, because this is a test."
    ):
        c.method()


def test_static_method_uninhibited() -> None:
    assert C.static_method(1) == 2
    with pytest.warns(expected_warning=QCoDeSDeprecationWarning):
        c = C("pristine")
        assert c.static_method(1) == 2


@pytest.mark.xfail(reason="This is not implemented yet.")
def test_static_method_raises(self) -> None:
    with assert_deprecated(
        "The function <static_method> is deprecated, because this is a test."
    ):
        assert C.static_method(1) == 2
    with pytest.warns(expected_warning=QCoDeSDeprecationWarning):
        c = C("pristine")
    with assert_deprecated(
        "The function <static_method> is deprecated, because this is a test."
    ):
        assert c.static_method(1) == 2


@pytest.mark.xfail(reason="This is not implemented yet.")
def test_class_method_uninhibited() -> None:
    with pytest.warns(expected_warning=QCoDeSDeprecationWarning):
        assert C.class_method(1) == 2
    with pytest.warns(expected_warning=QCoDeSDeprecationWarning):
        c = C("pristine")
    with pytest.warns(expected_warning=QCoDeSDeprecationWarning):
        assert c.class_method(1) == 2


@pytest.mark.xfail(reason="This is not implemented yet.")
def test_class_method_raises() -> None:
    with assert_deprecated(
        "The function <static_method> is deprecated, because this is a test."
    ):
        C.class_method(1)
    with pytest.warns(expected_warning=QCoDeSDeprecationWarning):
        c = C("pristine")
    with assert_deprecated(
        "The function <static_method> is deprecated, because this is a test."
    ):
        c.class_method(1)


def test_property_uninhibited() -> None:
    with pytest.warns(expected_warning=QCoDeSDeprecationWarning):
        c = C("pristine")
        assert c.prop == "pristine"


@pytest.mark.xfail(reason="This is not implemented yet.")
def test_property_raises() -> None:
    with pytest.warns(expected_warning=QCoDeSDeprecationWarning):
        c = C("pristine")
        with assert_deprecated(
            "The function <method> is deprecated, because this is a test."
        ):
            _ = c.prop


def test_setter_uninhibited() -> None:
    with pytest.warns(expected_warning=QCoDeSDeprecationWarning):
        c = C("pristine")
        c.prop = "changed"
        assert c.a == "changed_prop"


@pytest.mark.xfail(reason="This is not implemented yet.")
def test_setter_raises(self) -> None:
    with pytest.warns(expected_warning=QCoDeSDeprecationWarning):
        c = C("pristine")

    with assert_deprecated(
        "The function <method> is deprecated, because this is a test."
    ):
        c.prop = "changed"
