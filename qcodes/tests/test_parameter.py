"""
Test suite for parameter
"""
from collections import namedtuple
from unittest import TestCase

from qcodes import Function
from qcodes.instrument.parameter import (Parameter, ManualParameter,
                                         StandardParameter)
from qcodes.utils.validators import Numbers


class TestParamConstructor(TestCase):

    def test_name_s(self):
        p = Parameter('simple')
        self.assertEqual(p.name, 'simple')

        with self.assertRaises(ValueError):
            # you need a name of some sort
            Parameter()

        # or names
        names = ['H1', 'L1']
        p = Parameter(names=names)
        self.assertEqual(p.names, names)
        # if you don't provide a name, it's called 'None'
        # TODO: we should probably require an explicit name.
        self.assertEqual(p.name, 'None')

        # or both, that's OK too.
        names = ['Peter', 'Paul', 'Mary']
        p = Parameter(name='complex', names=names)
        self.assertEqual(p.names, names)
        # You can always have a name along with names
        self.assertEqual(p.name, 'complex')

        shape = (10,)
        setpoints = (range(10),)
        setpoint_names = ('my_sp',)
        setpoint_labels = ('A label!',)
        p = Parameter('makes_array', shape=shape, setpoints=setpoints,
                      setpoint_names=setpoint_names,
                      setpoint_labels=setpoint_labels)
        self.assertEqual(p.shape, shape)
        self.assertFalse(hasattr(p, 'shapes'))
        self.assertEqual(p.setpoints, setpoints)
        self.assertEqual(p.setpoint_names, setpoint_names)
        self.assertEqual(p.setpoint_labels, setpoint_labels)

        shapes = ((2,), (3,))
        setpoints = ((range(2),), (range(3),))
        setpoint_names = (('sp1',), ('sp2',))
        setpoint_labels = (('first label',), ('second label',))
        p = Parameter('makes arrays', shapes=shapes, setpoints=setpoints,
                      setpoint_names=setpoint_names,
                      setpoint_labels=setpoint_labels)
        self.assertEqual(p.shapes, shapes)
        self.assertFalse(hasattr(p, 'shape'))
        self.assertEqual(p.setpoints, setpoints)
        self.assertEqual(p.setpoint_names, setpoint_names)
        self.assertEqual(p.setpoint_labels, setpoint_labels)

    def test_repr(self):
        for i in [0, "foo", "", "fåil"]:
            with self.subTest(i=i):
                param = Parameter(name=i)
                s = param.__repr__()
                st = '<{}.{}: {} at {}>'.format(
                    param.__module__, param.__class__.__name__,
                    param.name, id(param))
                self.assertEqual(s, st)

    blank_instruments = (
        None,  # no instrument at all
        namedtuple('noname', '')(),  # no .name
        namedtuple('blank', 'name')('')  # blank .name
    )
    named_instrument = namedtuple('yesname', 'name')('astro')

    def test_full_name(self):
        # three cases where only name gets used for full_name
        for instrument in self.blank_instruments:
            p = Parameter(name='fred')
            p._instrument = instrument
            self.assertEqual(p.full_name, 'fred')

            p.name = None
            self.assertEqual(p.full_name, None)

        # and finally an instrument that really has a name
        p = Parameter(name='wilma')
        p._instrument = self.named_instrument
        self.assertEqual(p.full_name, 'astro_wilma')

        p.name = None
        self.assertEqual(p.full_name, None)

    def test_full_names(self):
        for instrument in self.blank_instruments:
            # no instrument
            p = Parameter(name='simple')
            p._instrument = instrument
            self.assertEqual(p.full_names, None)

            p = Parameter(names=['a', 'b'])
            p._instrument = instrument
            self.assertEqual(p.full_names, ['a', 'b'])

        p = Parameter(name='simple')
        p._instrument = self.named_instrument
        self.assertEqual(p.full_names, None)

        p = Parameter(names=['penn', 'teller'])
        p._instrument = self.named_instrument
        self.assertEqual(p.full_names, ['astro_penn', 'astro_teller'])

class TestManualParameter(TestCase):

    def test_bare_function(self):
        # not a use case we want to promote, but it's there...
        p = ManualParameter('test')

        def doubler(x):
            p.set(x * 2)

        f = Function('f', call_cmd=doubler, args=[Numbers(-10, 10)])

        f(4)
        self.assertEqual(p.get(), 8)
        with self.assertRaises(ValueError):
            f(20)

class TestStandardParam(TestCase):

     def test_param_cmd_with_parsing(self):
        def set_p(val):
            self._p = val

        def get_p():
            return self._p

        def parse_set_p(val):
            return '{:d}'.format(val)

        p = StandardParameter('p_int', get_cmd=get_p, get_parser=int,
                              set_cmd=set_p, set_parser=parse_set_p)

        p(5)
        self.assertEqual(self._p, '5')
        self.assertEqual(p(), 5)