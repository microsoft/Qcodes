from unittest import TestCase

from qcodes.instrument.parameter_node import ParameterNode
from qcodes.instrument.parameter import Parameter

class TestParameterNode(TestCase):
    def test_implicit_parameter_name(self):
        for use_as_attribute in [True, False]:
            parameter_node = ParameterNode(use_as_attributes=use_as_attribute)
            parameter_node.implicit_name_parameter = Parameter(set_cmd=None)

            # Test that name and label of parameter are implicitly set
            self.assertEqual(parameter_node['implicit_name_parameter'].name,
                             'implicit_name_parameter')
            self.assertEqual(parameter_node['implicit_name_parameter'].label,
                             'Implicit name parameter')

            parameter_node.explicit_name_parameter = Parameter(name='explicit',
                                                               set_cmd=None)
            # Test that name and label of parameter are not implicitly set
            self.assertEqual(parameter_node['explicit_name_parameter'].name,
                             'explicit')
            self.assertEqual(parameter_node['explicit_name_parameter'].label,
                             'explicit')

    def test_use_as_attributes(self):
        parameter_node = ParameterNode(use_as_attributes=True)
        test_parameter = Parameter(set_cmd=None)
        parameter_node.test_parameter = test_parameter

        self.assertIs(parameter_node['test_parameter'], test_parameter)

        self.assertEqual(parameter_node.test_parameter, None)
        parameter_node.test_parameter = 42
        self.assertEqual(parameter_node.test_parameter, 42)

    def test_use_not_as_attributes(self):
        parameter_node = ParameterNode(use_as_attributes=False)
        test_parameter = Parameter(set_cmd=None)
        parameter_node.test_parameter = test_parameter

        self.assertIs(parameter_node['test_parameter'], test_parameter)
        self.assertIs(parameter_node.test_parameter, test_parameter)

        self.assertEqual(parameter_node.test_parameter(), None)
        # Setting value of parameter still works via attribute
        parameter_node.test_parameter = 42
        self.assertEqual(parameter_node.test_parameter(), 42)

    def test_nested_parameter_node(self):
        parameter_node = ParameterNode(use_as_attributes=True)
        nested_parameter_node = ParameterNode(use_as_attributes=True)
        parameter_node.nested = nested_parameter_node
        self.assertEqual(parameter_node.nested.name, 'nested')

        # Add parameter
        parameter_node.nested.param = Parameter(set_cmd=None)
        parameter_node.nested.param = 42
        self.assertEqual(parameter_node.nested.param, 42)

    def test_parameter_node_nested_explicit_name(self):
        parameter_node = ParameterNode(use_as_attributes=True)
        nested_explicit_parameter_node = ParameterNode(name='explicit_name',
                                                       use_as_attributes=True)
        parameter_node.nested = nested_explicit_parameter_node
        self.assertEqual(parameter_node.nested.name, 'explicit_name')

