import pytest
import warnings

from qcodes.utils.deprecate import deprecate, issue_deprecation_warning


def test_issue_deprecation_warning():
    with warnings.catch_warnings(record=True) as w:
        issue_deprecation_warning(
            'use of this function',
            'of this being a test',
            'a real function'
        )
    assert issubclass(w[-1].category, DeprecationWarning)
    assert (str(w[-1].message) ==
            'The use of this function is deprecated, because '
            'of this being a test. Use \"a real function\" as an alternative.')


@pytest.mark.filterwarnings('ignore:The function "add_one" is deprecated,')
def test_similar_output():

    def _add_one(x):
        return 1 + x


    @deprecate(reason='this function is for private use only.')
    def add_one(x):
        return _add_one(x)

    assert add_one(1) == _add_one(1)
