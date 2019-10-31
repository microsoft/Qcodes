import pytest
import warnings

from qcodes.utils.deprecate import (
    deprecate, issue_deprecation_warning, QCoDeSDeprecationWarning,
    _catch_deprecation_warnings,
    assert_not_deprecated, assert_deprecated)


def test_issue_deprecation_warning():
    with assert_deprecated(
            'The use of this function is deprecated, because '
            'of this being a test. Use \"a real function\" as an alternative.'):
        issue_deprecation_warning(
            'use of this function',
            'of this being a test',
            'a real function'
        )

@pytest.mark.filterwarnings('ignore:The function "add_one" is deprecated,')
def test_similar_output():

    def _add_one(x):
        return 1 + x

    @deprecate(reason='this function is for private use only')
    def add_one(x):
        return _add_one(x)
    with assert_deprecated(
            'The function <add_one> is deprecated, because '
            'this function is for private use only.'):
        assert add_one(1) == _add_one(1)


def test_deprecated_context_manager():
    with _catch_deprecation_warnings() as ws:
        issue_deprecation_warning('something')
        issue_deprecation_warning('something more')
        warnings.warn('Some other warning')
    assert len(ws) == 2
    with warnings.catch_warnings(record=True) as ws:
        issue_deprecation_warning('something')
        warnings.warn('Some other warning')
    assert len(ws) == 2


def test_deprecated_class():

    @deprecate(reason='this is a test')
    class C:
        a = 'pristine'

        def method(self):
            self.a = 'last called by method'

        @staticmethod
        def static_method(a):
            return a + 1

    with assert_not_deprecated():
        c = C()
    with assert_deprecated(
            'The function <method> is deprecated, because '
            'this is a test.'):
        c.method()
    assert c.a == 'last called by method'

    with assert_deprecated(
            'The function <static_method> is deprecated, because '
            'this is a test.'):
        assert C.static_method(1) == 2
    assert c.a == 'last called by method'
