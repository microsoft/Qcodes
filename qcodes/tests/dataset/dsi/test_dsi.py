import re

import numpy as np
import pytest

from qcodes.dataset.data_storage_interface import DataWriterInterface


def test_validate_results_dict():
    """Test _validate_results_dict helper function of DSI"""
    validate = DataWriterInterface._validate_results_dict

    validate({'x': [1]})
    validate({'x': [1], 'y': [2]})

    validate({'x': [1, 2]})
    validate({'x': np.ndarray(1)})
    validate({'x': np.ndarray(2)})

    with pytest.raises(AssertionError):
        validate({})

    with pytest.raises(AssertionError):
        validate([])

    with pytest.raises(AssertionError):
        validate(())

    match_str = re.escape("'tuple' object has no attribute 'items'")
    with pytest.raises(AttributeError, match=match_str):
        validate((('x', 1), ('y', 2)))

    match_str = re.escape("object of type 'int' has no len()")
    with pytest.raises(TypeError, match=match_str):
        validate({'x': 1})

    with pytest.raises(AssertionError):
        validate({'x': []})

    with pytest.raises(AssertionError):
        validate({'x': [1], 'y': []})

    pytest.xfail('need more cases for validation function..')
