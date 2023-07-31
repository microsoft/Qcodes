from qcodes.utils import partial_with_docstring


def test_partial_with_docstring() -> None:
    def f():
        pass

    docstring = "some docstring"
    g = partial_with_docstring(f, docstring)
    assert g.__doc__ == docstring


def test_partial_with_docstring_pass_args() -> None:
    """
    When one uses partial to bind the last argument
    it should be possible to provide arguments before
    as positional args. This matches the behaviour of
    functools.partial
    """

    def f(a: int, b: int):
        return a + b

    docstring = "some docstring"
    g = partial_with_docstring(f, docstring, b=1)

    assert g.__doc__ == docstring
    assert g(2) == 3


def test_partial_with_docstring_returns_value() -> None:
    def f(a: int, b: int):
        return a + b

    docstring = "some docstring"
    g = partial_with_docstring(f, docstring, a=1)

    assert g.__doc__ == docstring
    assert g(b=2) == 3
