from qcodes.utils.helpers import partial_with_docstring


def test_partial_with_docstring():
    def f():
        pass

    docstring = "some docstring"
    g = partial_with_docstring(f, docstring)
    assert g.__doc__ == docstring


def test_partial_with_docstring_returns_value():
    def f(a: int, b: int):
        return a + b

    docstring = "some docstring"
    g = partial_with_docstring(f, docstring, a=1)

    assert g.__doc__ == docstring
    assert g(2) == 3


def test_partial_with_docstring_returns_value_2():
    def f(a: int, b: int):
        return a + b

    docstring = "some docstring"
    g = partial_with_docstring(f, docstring, a=1)

    assert g.__doc__ == docstring
    assert g(b=2) == 3
