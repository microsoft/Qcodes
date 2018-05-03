from unittest import TestCase
from copy import copy, deepcopy

from qcodes.instrument.parameter_node import ParameterNode, parameter
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

    def test_parameter_node_decorator(self):
        class Node(ParameterNode):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.val = 42
                self.param1 = Parameter()

            @parameter
            def param1_get(self, parameter):
                self.get_accessed = True
                parameter.get_accessed = True

                return self.val

            @parameter
            def param1_set(self, parameter, val):
                self.set_accessed = True
                parameter.set_accessed = True
                self.val = val

        node = Node('node', use_as_attributes=True)
        self.assertIn('param1', node.parameters)
        self.assertFalse(hasattr(node, 'get_accessed'))
        self.assertFalse(hasattr(node['param1'], 'get_accessed'))
        self.assertFalse(hasattr(node, 'set_accessed'))
        self.assertFalse(hasattr(node['param1'], 'set_accessed'))

        self.assertEqual(node.param1, 42)
        self.assertTrue(hasattr(node, 'get_accessed'))
        self.assertTrue(hasattr(node['param1'], 'get_accessed'))

        node.param1 = 32
        self.assertEqual(node.param1, 32)
        self.assertTrue(hasattr(node, 'set_accessed'))
        self.assertTrue(hasattr(node['param1'], 'set_accessed'))

    def test_parameter_node_decorator_validator(self):
        class Node(ParameterNode):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.val = 42
                self.param1 = Parameter()

            @parameter
            def param1_get(self, parameter):
                self.get_accessed = True
                parameter.get_accessed = True

                return self.val

            @parameter
            def param1_set(self, parameter, val):
                self.set_accessed = True
                parameter.set_accessed = True
                self.val = val

            @parameter
            def param1_vals(self, parameter, val):
                return val > 32

        node = Node('node', use_as_attributes=True)
        with self.assertRaises(ValueError):
            node.param1 = 31
        node.param1 = 44

class TestCopyParameterNode(TestCase):
    def test_copy_parameter_node(self):
        node = ParameterNode(use_as_attributes=True)
        node.p = Parameter(set_cmd=None)
        node.p = 123

        node2 = copy(node)
        self.assertEqual(node.p, 123)
        self.assertEqual(node2.p, 123)

        node3 = deepcopy(node)
        self.assertEqual(node3.p, 123)

        node.p = 124
        self.assertEqual(node.p, 124)
        self.assertEqual(node2.p, 123)
        self.assertEqual(node3.p, 123)

        node2.p = 125
        self.assertEqual(node.p, 124)
        self.assertEqual(node2.p, 125)
        self.assertEqual(node3.p, 123)

        node3.p = 126
        self.assertEqual(node.p, 124)
        self.assertEqual(node2.p, 125)
        self.assertEqual(node3.p, 126)

    def test_copy_parameter_in_node(self):
        node = ParameterNode(use_as_attributes=True)
        node.p = Parameter(set_cmd=None)

        node.p = 123

        p_copy = copy(node['p'])
        self.assertEqual(node.p, 123)
        self.assertEqual(p_copy(), 123)

        node.p = 124
        self.assertEqual(node.p, 124)
        self.assertEqual(p_copy(), 123)

        p_copy(125)
        self.assertEqual(node.p, 124)
        self.assertEqual(p_copy(), 125)

    def test_copy_parameter_node_with_parameter_decorator(self):
        class Node(ParameterNode):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.p = Parameter()
                self.value = 1

            @parameter
            def p_get(self, parameter):
                return (self.value, parameter._value)

            @parameter
            def p_set(self, parameter, value):
                parameter._value = value

            @parameter
            def p_vals(self, parameter, val):
                return True

        node = Node(use_as_attributes=True)

        node.p = 42
        self.assertEqual(node.p, (1,42))

        node_copy = copy(node)
        node_copy.name = 'node_copy'
        node_copy.value = 2
        self.assertEqual(node.p, (1,42))
        self.assertEqual(node_copy.p, (2, 42))

        node.p = 43
        self.assertEqual(node.p, (1, 43))
        self.assertEqual(node_copy.p, (2, 42))

        node_copy.p = 44
        self.assertEqual(node.p, (1, 43))
        self.assertEqual(node_copy.p, (2, 44))

    def test_deepcopy_parameter_node_with_parameter_decorator(self):
        class Node(ParameterNode):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.p = Parameter()
                self.value = 1

            @parameter
            def p_get(self, parameter):
                return (self.value, parameter._value)

            @parameter
            def p_set(self, parameter, value):
                parameter._value = value

            @parameter
            def p_vals(self, parameter, val):
                return True

        node = Node(use_as_attributes=True)

        node.p = 42
        self.assertEqual(node.p, (1,42))

        node_copy = copy(node)
        node_copy.value = 2
        self.assertEqual(node.p, (1,42))
        self.assertEqual(node_copy.p, (2, 42))

        node.p = 43
        self.assertEqual(node.p, (1, 43))
        self.assertEqual(node_copy.p, (2, 42))

        node_copy.p = 44
        self.assertEqual(node.p, (1, 43))
        self.assertEqual(node_copy.p, (2, 44))

    def test_copy_parameter_in_node_with_decorator(self):
        class Node(ParameterNode):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.p = Parameter()
                self.value = 1

            @parameter
            def p_get(self, parameter):
                return (self.value, parameter._value)

            @parameter
            def p_set(self, parameter, value):
                parameter._value = value

            @parameter
            def p_vals(self, parameter, val):
                return True

        node = Node(use_as_attributes=True)

        node.p = 42
        self.assertEqual(node.p, (1,42))

        parameter_copy = copy(node['p'])
        self.assertEqual(parameter_copy(), (1, 42))

        node.value = 2
        self.assertEqual(node.p, (2, 42))
        self.assertEqual(parameter_copy(), (2, 42))

        node.p = 43
        self.assertEqual(node.p, (2, 43))
        self.assertEqual(parameter_copy(), (2, 42))

        parameter_copy(44)
        self.assertEqual(node.p, (2, 43))
        self.assertEqual(parameter_copy(), (2, 44))

    def test_deepcopy_parameter_in_node_with_decorator(self):
        class Node(ParameterNode):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.p = Parameter()
                self.value = 1

            @parameter
            def p_get(self, parameter):
                return (self.value, parameter._value)

            @parameter
            def p_set(self, parameter, value):
                parameter._value = value

            @parameter
            def p_vals(self, parameter, val):
                return True

        node = Node(use_as_attributes=True)

        node.p = 42
        self.assertEqual(node.p, (1,42))

        parameter_copy = deepcopy(node['p'])
        self.assertEqual(parameter_copy(), (1, 42))

        node.value = 2
        self.assertEqual(node.p, (2, 42))
        self.assertEqual(parameter_copy(), (2, 42))

        node.p = 43
        self.assertEqual(node.p, (2, 43))
        self.assertEqual(parameter_copy(), (2, 42))

        parameter_copy(44)
        self.assertEqual(node.p, (2, 43))
        self.assertEqual(parameter_copy(), (2, 44))
