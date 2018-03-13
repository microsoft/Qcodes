import operator
import warnings
from qcodes.utils.helpers import is_function


class DeferredOperations:
    """
    NOTE: This class is DEPRECATED and you should not use it. You can
    achieve the same functionality using good ole lambdas.

    Make math and logic operations return callables to defer execution
    of arbitrary formulae.

    Can be used as a mixin class - in which case you should NOT call __init__
    and you SHOULD override __call__ and get with identical results.


    You can evaluate a DeferredOperations by calling it or
    by calling its .get() method

    Args:

        call_func: a callable taking no arguments
        call_parts: the original callables, in case we want to
            record them all (ie raw data)

    Examples:
        >>> d = DeferredOperations(lambda:42)
        >>> d()
        42
        >>> (d*5)()
        210
        >>> (d>10)()
        True
        >>> ((84/d) + (d*d))()
        1766

    Note:
        and & or are special: there are no magic methods for
        the regular and & or, so we need to use the bitwise operators &,| as
        boolean operators. We DO NOT short-circuit them; the right side is
        always evaluated.
    """

    warnings.warn("DeferredOperations has been deprecated and will be removed "
                  "in the not-too-distant future.")

    def __init__(self, call_func, args=(), call_parts=()):
        self._validate_callable(call_func, len(args))
        self.call_func = call_func
        self.args = args
        self.call_parts = call_parts

        # only bind self.get if we explicitly initialize the object as a
        # DeferredOperations, so that an object without a `get` method
        # that simply inherits DeferredOperations behavior will not
        # accidentally look gettable.
        self.get = self.__call__

    def __call__(self):
        return self.call_func(*self.args)

    def __bool__(self):
        raise TypeError('This is a DeferredOperations object, you must '
                        'call or .get() it before testing its truthiness',
                        self)

    def _validate_callable(self, func, arg_count=0):
        if not is_function(func, arg_count):
            raise TypeError(
                'function must be a callable taking '
                '{} arguments, not {}'.format(arg_count, repr(func)))

    def _call_unary(self, op):
        return op(self())

    def _unary(self, op):
        if not getattr(self, 'call_parts', None):
            self.call_parts = (self,)

        return DeferredOperations(self._call_unary, (op,), self.call_parts)

    def _call_binary_callable(self, op, other):
        return op(self(), other())

    def _call_binary_constant(self, op, other):
        return op(self(), other)

    def _binary(self, op, other):
        if not getattr(self, 'call_parts', None):
            self.call_parts = (self,)

        if callable(other):
            self._validate_callable(other)
            other_parts = getattr(other, 'call_parts', (other,))
            return DeferredOperations(self._call_binary_callable, (op, other),
                                      self.call_parts + other_parts)

        # TODO: any other types we should support?
        elif isinstance(other, (int, float, str, type(None))):
            return DeferredOperations(self._call_binary_constant, (op, other),
                                      self.call_parts)
        else:
            raise TypeError(
                'DeferredOperations are not compatible with '
                '{}, type {}'.format(repr(other), repr(type(other))))

    def __eq__(self, other):
        if id(self) == id(other):
            return True
        return self._binary(operator.eq, other)

    def __ne__(self, other):
        return self._binary(operator.ne, other)

    def __ge__(self, other):
        return self._binary(operator.ge, other)

    def __gt__(self, other):
        return self._binary(operator.gt, other)

    def __le__(self, other):
        return self._binary(operator.le, other)

    def __lt__(self, other):
        return self._binary(operator.lt, other)

    def __abs__(self):
        return self._unary(operator.abs)

    def __add__(self, other):
        return self._binary(operator.add, other)

    def __and__(self, other):
        """
        uses the bitwise and operator & to do boolean and
        """
        return self._binary(_and, other)

    def __floordiv__(self, other):
        return self._binary(operator.floordiv, other)

    def __mod__(self, other):
        return self._binary(operator.mod, other)

    def __mul__(self, other):
        return self._binary(operator.mul, other)

    def __neg__(self):
        return self._unary(operator.neg)

    def __or__(self, other):
        """
        uses the bitwise or operator | to do boolean or
        """
        return self._binary(_or, other)

    def __pos__(self):
        return self._unary(operator.pos)

    def __pow__(self, other):
        return self._binary(operator.pow, other)

    def __sub__(self, other):
        return self._binary(operator.sub, other)

    def __truediv__(self, other):
        return self._binary(operator.truediv, other)

    def __radd__(self, other):
        return self._binary(operator.add, other)

    def __rsub__(self, other):
        return self._binary(_rsub, other)

    def __rmul__(self, other):
        return self._binary(operator.mul, other)

    def __rtruediv__(self, other):
        return self._binary(_rtruediv, other)

    def __rfloordiv__(self, other):
        return self._binary(_rfloordiv, other)

    def __rmod__(self, other):
        return self._binary(_rmod, other)

    def __rpow__(self, other):
        return self._binary(_rpow, other)

    def __rand__(self, other):
        return self._binary(_rand, other)

    def __ror__(self, other):
        return self._binary(_ror, other)

    def __round__(self, other=None):
        if other is None:
            return self._unary(round)
        else:
            return self._binary(round, other)


# functional forms not in the operator module, so we get
# the order of arguments correct

def _and(a, b):
    return a and b


def _rand(a, b):
    return b and a


def _or(a, b):
    return a or b


def _ror(a, b):
    return b or a


def _rsub(a, b):
    return b - a


def _rtruediv(a, b):
    return b / a


def _rfloordiv(a, b):
    return b // a


def _rmod(a, b):
    return b % a


def _rpow(a, b):
    return b ** a
