from unittest import TestCase

from qcodes.utils.workarounds import _null_context_manager_with_arguments
from qcodes.utils.workarounds import _visa_resource_read_termination_set_to_none
from qcodes.utils.workarounds import visa_query_binary_values_fix_for


class TestVisaQueryBinaryValuesFixContextManager(TestCase):
    """
    Test visa_query_binary_values_fix_for context manager
    """

    class VisaResourceMock:
        _read_termination = '\n'

    def test_context_manager_choice(self):
        import visa
        if visa.__version__ == '1.9.0':
            assert _visa_resource_read_termination_set_to_none \
                   is visa_query_binary_values_fix_for
        else:
            assert _null_context_manager_with_arguments \
                   is visa_query_binary_values_fix_for

    def test_for_pyvisa_1_9_0(self):
        """Test for pyvisa 1.9.0 that context manager alters read termination"""

        resource = self.VisaResourceMock()
        assert '\n' == resource._read_termination

        with _visa_resource_read_termination_set_to_none(resource):
            assert None is resource._read_termination

        assert '\n' == resource._read_termination

    def test_with_pyvisa_1_8(self):
        """Test for pyvisa 1.8 that the context manager is null"""

        resource = self.VisaResourceMock()
        assert '\n' == resource._read_termination

        with _null_context_manager_with_arguments(resource):
            assert '\n' == resource._read_termination

        assert '\n' == resource._read_termination
