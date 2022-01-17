from qcodes.utils.helpers import partial_with_docstring


def test_partial_with_docstring():
    def f():
        pass

    docstring = "some docstring"
    g = partial_with_docstring(f, docstring)
    assert g.__doc__ == docstring
