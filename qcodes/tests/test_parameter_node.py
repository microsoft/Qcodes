from unittest import TestCase
from copy import copy, deepcopy
import pickle

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
                             'Explicit')

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
        self.assertEqual(parameter_node.nested.parent, parameter_node)
        self.assertNotIn('parent', parameter_node.parameter_nodes)

        # Add parameter
        parameter_node.nested.param = Parameter(set_cmd=None)
        parameter_node.nested.param = 42
        self.assertEqual(parameter_node.nested.param, 42)

    def test_nested_parameter_node_no_parent(self):
        parameter_node = ParameterNode(use_as_attributes=True)
        nested_parameter_node = ParameterNode(use_as_attributes=True)
        nested_parameter_node.parent = False # Should not have parent
        parameter_node.nested = nested_parameter_node

        self.assertEqual(parameter_node.nested.name, 'nested')
        self.assertEqual(parameter_node.nested.parent, False)
        self.assertNotIn('parent', parameter_node.parameter_nodes)

    def test_nested_parameter_node_snapshot_with_parent(self):
        parameter_node = ParameterNode(use_as_attributes=True)
        nested_parameter_node = ParameterNode(use_as_attributes=True)
        parameter_node.nested = nested_parameter_node

        nested_parameter_node.parent = parameter_node
        nested_parameter_node.snapshot()

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

    def test_parameter_node_subclass_decorator(self):
        class Node(ParameterNode):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.val = 42
                self.param1 = Parameter()

            @parameter
            def param1_get(self, parameter):
                return self.val

        class SubNode(Node):
            pass

        subnode = SubNode()
        self.assertEqual(subnode.param1(), 42)

    def test_sweep_parameter_node(self):
        node = ParameterNode(use_as_attributes=True)
        node.param1 = Parameter(set_cmd=None)
        sweep_values_node = node.sweep('param1', 0, 10, step=1)
        sweep_values_parameter = node['param1'].sweep(0, 10, step=1)
        self.assertEqual(list(sweep_values_node), list(sweep_values_parameter))

    def test_parameter_str(self):
        p = Parameter(set_cmd=None)
        self.assertEqual(str(p), 'None')
        p = Parameter('param1', set_cmd=None)
        self.assertEqual(str(p), 'param1')

        node = ParameterNode()
        p.parent = node
        self.assertEqual(str(p), 'param1')
        node.name = 'node1'
        self.assertEqual(str(p), 'node1_param1')

    def test_parameter_no_parent(self):
        p = Parameter(set_cmd=None)
        p.parent = False

        node = ParameterNode()
        node.p = p
        self.assertIs(p.parent, False)


class TestCopyParameterNode(TestCase):
    def test_copy_parameter_node(self):
        node = ParameterNode(use_as_attributes=True)
        node.p = Parameter(set_cmd=None)
        node.p = 123
        self.assertEqual(node['p'].parent, node)

        node2 = copy(node)
        self.assertEqual(node2['p'].parent, node2)
        self.assertIsNot(node['p'], node2['p'])
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

    def test_deepcopy_parameter_node(self):
        node = ParameterNode(use_as_attributes=True)
        node.p = Parameter(set_cmd=None)
        node.p = 123
        self.assertEqual(node['p'].parent, node)

        node2 = deepcopy(node)
        self.assertEqual(node2['p'].parent, None)
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

    def test_copy_copied_parameter_node(self):
        node = ParameterNode(use_as_attributes=True)
        node.p = Parameter(set_cmd=None)
        node.p = 123
        self.assertEqual(node['p'].parent, node)

        node2 = copy(node)
        node3 = copy(node2)
        self.assertEqual(node2['p'].parent, node2)
        self.assertEqual(node3['p'].parent, node3)

        self.assertIsNot(node['p'], node2['p'])
        self.assertIsNot(node2['p'], node3['p'])
        self.assertEqual(node.p, 123)
        self.assertEqual(node2.p, 123)
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

    def test_copy_parameter_node_add_parameter(self):
        node = ParameterNode(use_as_attributes=True)
        node.add_parameter('p', set_cmd=None)
        node.p = 123
        self.assertEqual(node['p'](), 123)
        self.assertEqual(node['p']._instrument, node)

        node_copy = copy(node)
        self.assertEqual(node.p, 123)
        self.assertEqual(node['p']._instrument, node)
        self.assertEqual(node_copy['p']._instrument, node_copy)

    def test_deepcopy_parameter_node_add_parameter(self):
        node = ParameterNode(use_as_attributes=True)
        node.add_parameter('p', set_cmd=None)
        node.p = 123
        self.assertEqual(node['p'](), 123)
        self.assertEqual(node['p']._instrument, node)

        node_copy = deepcopy(node)
        self.assertEqual(node.p, 123)
        self.assertEqual(node['p']._instrument, node)
        self.assertEqual(node_copy['p']._instrument, None)

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

    def test_deepcopy_parameter_in_node(self):
        node = ParameterNode(use_as_attributes=True)
        node.p = Parameter(set_cmd=None)

        node.p = 123

        p_copy = deepcopy(node['p'])
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

        node_copy = deepcopy(node)
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

    def test_copy_parameter_node_with_parameter_decorator_2(self):
        # Create node whose parameter returns node and parameter objects
        class Node(ParameterNode):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.p = Parameter()
                self.value = 1

            @parameter
            def p_get(self, parameter):
                return (self, parameter, parameter._value)

            @parameter
            def p_set(self, parameter, value):
                parameter._value = value

        node = Node(use_as_attributes=True)
        node_copy = copy(node)
        self.assertIsNot(node['p'], node_copy['p'])

        node_copy.p = 2
        returned_node, returned_parameter, returned_val = node_copy.p
        self.assertIs(returned_node, node_copy)
        self.assertIs(returned_parameter, node_copy['p'])
        self.assertIs(returned_val, 2)

    def test_deepcopy_parameter_node_with_parameter_decorator_2(self):
        # Create node whose parameter returns node and parameter objects
        class Node(ParameterNode):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.p = Parameter()
                self.value = 1

            @parameter
            def p_get(self, parameter):
                return (self, parameter, parameter._value)

            @parameter
            def p_set(self, parameter, value):
                parameter._value = value

        node = Node(use_as_attributes=True)
        node_copy = deepcopy(node)
        self.assertIsNot(node['p'], node_copy['p'])

        node_copy.p = 2
        returned_node, returned_parameter, returned_val = node_copy.p
        self.assertIs(returned_node, node_copy)
        self.assertIs(returned_parameter, node_copy['p'])
        self.assertIs(returned_val, 2)

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

    def test_copy_nested_node(self):
        node = ParameterNode('node')
        nested_node = ParameterNode('nested_node')

        node.nested_node = nested_node
        self.assertEqual(node.nested_node.parent, node)

        copied_nested_node = copy(nested_node)
        self.assertEqual(copied_nested_node.parent, None)

    def test_deepcopy_nested_node(self):
        node = ParameterNode('node')
        nested_node = ParameterNode('nested_node')

        node.nested_node = nested_node
        self.assertEqual(node.nested_node.parent, node)

        copied_nested_node = deepcopy(nested_node)
        self.assertEqual(copied_nested_node.parent, None)


class ParameterAndNode(Parameter, ParameterNode):
    def __init__(self, **kwargs):
        Parameter.__init__(self, **kwargs)
        ParameterNode.__init__(self, use_as_attributes=True, **kwargs)
        self.value = 42

    def get_raw(self):
        return self.value

    def set_raw(self, value):
        self.value = value


class TestCombinedParameterAndParameterNode(TestCase):
    def test_overlapping_attributes(self):
        # If any new attributes get introduced to either, this test will check
        # if it overlaps.
        node = ParameterNode('node')
        p = Parameter('param', set_cmd=None)

        parameter_dict = [*p.__dict__, *Parameter.__dict__]
        node_dict = [*node.__dict__, *ParameterNode.__dict__]
        overlapping_attrs = {k for k in parameter_dict + node_dict
                             if k in parameter_dict and k in node_dict}
        # Remove slotnames if it exists, used for caching __getstate__
        try:
            overlapping_attrs.remove('__slotnames__')
        except:
            pass

        self.assertSetEqual(overlapping_attrs,
                            {'__init__', '_meta_attrs', '__doc__', '__module__',
                             'metadata', '__deepcopy__', 'name', '__getitem__',
                             'log_changes', 'sweep', 'parent', 'get'})

    def test_create_multiple_inheritance_initialization(self):
        class ParameterAndNode(Parameter, ParameterNode):
            def __init__(self, **kwargs):
                Parameter.__init__(self, **kwargs)
                ParameterNode.__init__(self, use_as_attributes=True, **kwargs)

        parameter_and_node = ParameterAndNode()

        # Check if ParameterNode instance attribute exists
        # which means it went through its __init__
        self.assertTrue(hasattr(parameter_and_node, 'functions'))
        # Same for parameter
        self.assertTrue(hasattr(parameter_and_node, 'step'))

    def test_parameter_get_set(self):

        parameter_and_node = ParameterAndNode()
        self.assertEqual(parameter_and_node(), 42)

        parameter_and_node(41)
        self.assertEqual(parameter_and_node(), 41)

    def test_add_parameter_to_node(self):
        parameter_and_node = ParameterAndNode()
        p = Parameter('p', initial_value=42)
        parameter_and_node.p = p
        self.assertEqual(parameter_and_node.p, 42)
        self.assertEqual(parameter_and_node['p'], p)

        self.assertEqual(parameter_and_node(), 42)
        parameter_and_node(43)
        self.assertEqual(parameter_and_node(), 43)

    def test_connect_parameter_and_node_to_parameter(self):
        parameter_and_node = ParameterAndNode()
        p = Parameter('p', initial_value=0, set_cmd=None)
        parameter_and_node.p = p

        p_source = Parameter('p', initial_value=42, set_cmd=None)
        p_source.connect(parameter_and_node['p'], update=True)
        self.assertEqual(parameter_and_node.p, 42)

        p_source(40)
        self.assertEqual(parameter_and_node.p, 40)

    def test_connect_parameter_to_parameter_and_node(self):
        parameter_and_node = ParameterAndNode()
        p = Parameter('p', initial_value=0, set_cmd=None)
        parameter_and_node.p = p

        p_target = Parameter('p', initial_value=42, set_cmd=None)
        parameter_and_node['p'].connect(p_target, update=True)
        self.assertEqual(p_target(), 0)

        parameter_and_node.p = 40
        self.assertEqual(p_target(), 40)

    def test_copy_parameter_and_node(self):
        parameter_and_node = ParameterAndNode()
        p = Parameter('p', initial_value=42, set_cmd=None)
        parameter_and_node.p = p

        copy_parameter_and_node = deepcopy(parameter_and_node)
        self.assertEqual(copy_parameter_and_node.p, 42)

    def test_copy_connected_parameter_and_node(self):
        parameter_and_node = ParameterAndNode()
        p = Parameter('p', initial_value=0, set_cmd=None)
        parameter_and_node.p = p

        p_source = Parameter('p', initial_value=42, set_cmd=None)
        p_source.connect(parameter_and_node['p'], update=True)
        self.assertEqual(parameter_and_node.p, 42)

        copy_parameter_and_node = deepcopy(parameter_and_node)
        self.assertEqual(copy_parameter_and_node.p, 42)

        p_source(41)
        self.assertEqual(parameter_and_node.p, 41)
        self.assertEqual(copy_parameter_and_node.p, 42)


class TestParameterNodeLogging(TestCase):
    def test_empty_node_snapshot(self):
        node = ParameterNode()
        snapshot = node.snapshot()
        self.assertCountEqual(snapshot.keys(),
                             ['__class__', 'functions', 'parameters',
                              'submodules', 'parameter_nodes'])

        node.name = 'node_name'
        self.assertEqual(node.snapshot()['name'], 'node_name')

    def test_parameter_in_node_snapshot(self):
        node = ParameterNode()
        node.p = Parameter()
        self.assertEqual(node.snapshot()['parameters']['p'], node.p.snapshot())

    def test_parameter_in_node_simplified_snapshot(self):
        node = ParameterNode(simplify_snapshot=True)
        node.p = Parameter(initial_value=42)
        self.assertEqual(node.snapshot()['p'], 42)

    def test_subnode_snapshot(self):
        node = ParameterNode()
        node.subnode = ParameterNode()

        self.assertDictEqual(node.snapshot()['parameter_nodes']['subnode'],
                             node.subnode.snapshot())

        node.subnode.param = Parameter()
        self.assertDictEqual(node.snapshot()['parameter_nodes']['subnode'],
                             node.subnode.snapshot())


class TestParameterNodePickling(TestCase):
    def test_pickled_empty_node(self):
        node = ParameterNode(name='node')
        pickle_dump = pickle.dumps(node)

        node_pickled = pickle.loads(pickle_dump)
        self.assertEqual(node_pickled.name, 'node')

    def test_pickled_node_with_parameter(self):
        node = ParameterNode(name='node')
        node.p = Parameter(initial_value=123)
        pickle_dump = pickle.dumps(node)

        node_pickled = pickle.loads(pickle_dump)
        self.assertEqual(node_pickled.name, 'node')
        self.assertEqual(node_pickled['p'].name, 'p')
        self.assertEqual(node_pickled['p'].get_latest(), 123)
        self.assertEqual(node_pickled['p'].get(), 123)


    def test_pickled_node_with_decorated_parameter(self):
        node = DecoratedNode(name='node')
        pickle_dump = pickle.dumps(node)

        node_pickled = pickle.loads(pickle_dump)
        self.assertEqual(node_pickled.name, 'node')
        self.assertEqual(node_pickled['p'].name, 'p')
        self.assertEqual(node_pickled['p'].get(), None)


class DecoratedNode(ParameterNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.p = Parameter()

    @parameter
    def p_get(self):
        return 42