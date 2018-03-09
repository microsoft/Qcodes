import math
from typing import Union, Tuple, cast, Optional

import numpy as np

BIGSTRING = 1000000000
BIGINT = int(1e18)


def validate_all(*args, context=''):
    """
    Takes a list of (validator, value) couplets and tests whether they are
    all valid, raising ValueError otherwise

    context: keyword-only arg with a string to include in the error message
        giving the user context for the error
    """
    if context:
        context = '; ' + context

    for i, (validator, value) in enumerate(args):
        validator.validate(value, 'argument ' + str(i) + context)


def range_str(min_val, max_val, name):
    """
    utility to represent ranges in Validator repr's
    """
    if max_val is not None:
        if min_val is not None:
            if max_val == min_val:
                return ' {}={}'.format(name, min_val)
            else:
                return ' {}<={}<={}'.format(min_val, name, max_val)
        else:
            return ' {}<={}'.format(name, max_val)
    elif min_val is not None:
        return ' {}>={}'.format(name, min_val)
    else:
        return ''


class Validator:
    """
    base class for all value validators
    each should have its own constructor, and override:

    __init__: here a private attribute, _valid_values, should be set.
        _valid_values must be a list of at least one valid value.
        If possible, it should include all valid values. The purpose of
        this attribute is to make it possible to find a valid value for
        a Parameter, given its validator.

    validate: function of two args: value, context
        value is what you're testing
        context is a string identifying the caller better

        raises an error (TypeError or ValueError) if the value fails

    valid_values: a property exposing _valid_values, which is a list
        of examples of valid values. For very simple validators, like
        Bool or Enum, the list contains all valid values, but in general
        it just holds SOME valid values. These example values are intended
        to be useful when simulating instruments.

    is_numeric: is this a numeric type (so it can be swept)?
    """

    def __init__(self):
        raise NotImplementedError

    def validate(self, value, context=''):
        raise NotImplementedError

    @property
    def valid_values(self):
        if not hasattr(self, '_valid_values'):
            raise NotImplementedError
        else:
            return self._valid_values

    is_numeric = False  # is this a numeric type (so it can be swept)?


class Anything(Validator):
    """allow any value to pass"""

    def __init__(self):
        self._valid_values = [0]

    def validate(self, value, context=''):
        pass
    # NOTE(giulioungaretti): why is_numeric?
    # it allows for set_step in parameter
    # TODO(giulioungaretti): possible refactor
    is_numeric = True

    def __repr__(self):
        return '<Anything>'

class Nothing(Validator):
    """allow no value to pass"""

    def __init__(self, reason):
        if reason:
            self.reason = reason
        else:
            self.reason = "Nothing Validator"

    def validate(self, value, context=''):
        raise RuntimeError("{}; {}".format(self.reason, context))

    def __repr__(self):
        return '<Nothing({})>'.format(self.reason)

class Bool(Validator):
    """
    requires a boolean
    """

    def __init__(self):
        self._valid_values = [True, False]

    def validate(self, value, context=''):
        if not isinstance(value, bool) and not isinstance(value, np.bool8):
            raise TypeError(
                '{} is not Boolean; {}'.format(repr(value), context))

    def __repr__(self):
        return '<Boolean>'


class Strings(Validator):
    """
    requires a string
    optional parameters min_length and max_length limit the allowed length
    to min_length <= len(value) <= max_length
    """

    def __init__(self, min_length=0, max_length=BIGSTRING):
        if isinstance(min_length, int) and min_length >= 0:
            self._min_length = min_length
        else:
            raise TypeError('min_length must be a non-negative integer')
        if isinstance(max_length, int) and max_length >= max(min_length, 1):
            self._max_length = max_length
        else:
            raise TypeError('max_length must be a positive integer '
                            'no smaller than min_length')
        self._valid_values = ['.'*min_length]

    def validate(self, value, context=''):
        if not isinstance(value, str):
            raise TypeError(
                '{} is not a string; {}'.format(repr(value), context))

        vallen = len(value)
        if vallen < self._min_length or vallen > self._max_length:
            raise ValueError(
                '{} is invalid: length must be between '
                '{} and {} inclusive; {}'.format(
                    repr(value), self._min_length, self._max_length, context))

    def __repr__(self):
        minv = self._min_length or None
        maxv = self._max_length if self._max_length < BIGSTRING else None
        return '<Strings{}>'.format(range_str(minv, maxv, 'len'))


class Numbers(Validator):
    """
    Args:
        min_value: Minimal value allowed, default -inf
        max_value: Maximal value allowed, default inf

    Raises:

    Todo:
        - fix raises
    """

    validtypes = (float, int, np.integer, np.floating)

    def __init__(self, min_value: Union[int, float]=-float("inf"),
                 max_value: Union[int, float]=float("inf")) -> None:

        if isinstance(min_value, self.validtypes):
            self._min_value = min_value
        else:
            raise TypeError('min_value must be a number')

        valuesok = max_value > min_value  # type: ignore
        valuesok = cast(bool, valuesok)

        if isinstance(max_value, self.validtypes) and valuesok:
            self._max_value = max_value
        else:
            raise TypeError('max_value must be a number bigger than min_value')

        self._valid_values = [min_value, max_value]

    def validate(self, value, context=''):
        if not isinstance(value, self.validtypes):
            raise TypeError(
                '{} is not an int or float; {}'.format(repr(value), context))

        if not (self._min_value <= value <= self._max_value):
            raise ValueError(
                '{} is invalid: must be between '
                '{} and {} inclusive; {}'.format(
                    repr(value), self._min_value, self._max_value, context))

    is_numeric = True

    def __repr__(self):
        minv = self._min_value if math.isfinite(self._min_value) else None
        maxv = self._max_value if math.isfinite(self._max_value) else None
        return '<Numbers{}>'.format(range_str(minv, maxv, 'v'))


class Ints(Validator):
    """
    requires an integer
    optional parameters min_value and max_value enforce
    min_value <= value <= max_value
    """

    validtypes = (int, np.integer)

    def __init__(self, min_value=-BIGINT, max_value=BIGINT):
        if isinstance(min_value, self.validtypes):
            self._min_value = int(min_value)
        else:
            raise TypeError('min_value must be an integer')

        if not isinstance(max_value, self.validtypes):
            raise TypeError('max_value must be an integer')
        if max_value > min_value:
            self._max_value = int(max_value)
        else:
            raise TypeError(
                'max_value must be an integer bigger than min_value')

        self._valid_values = [min_value, max_value]

    def validate(self, value, context=''):
        if not isinstance(value, self.validtypes):
            raise TypeError(
                '{} is not an int; {}'.format(repr(value), context))

        if not (self._min_value <= value <= self._max_value):
            raise ValueError(
                '{} is invalid: must be between '
                '{} and {} inclusive; {}'.format(
                    repr(value), self._min_value, self._max_value, context))

    is_numeric = True

    def __repr__(self):
        minv = self._min_value if self._min_value > -BIGINT else None
        maxv = self._max_value if self._max_value < BIGINT else None
        return '<Ints{}>'.format(range_str(minv, maxv, 'v'))


class PermissiveInts(Ints):
    """
    requires an integer or a float close to an integer
    optional parameters min_value and max_value enforce
    min_value <= value <= max_value
    Note that you probably always want to use this with a
    set_parser that converts the float repr to an actual int
    """

    def validate(self, value, context=''):
        if isinstance(value, (float, np.floating)):
            intrepr = int(round(value))
            remainder = abs(value - intrepr)
            if remainder < 1e-05:
                castvalue = intrepr
            else:
                raise TypeError('{} is not an int or close to an int'
                                '; {}'.format(repr(value), context))
        else:
            castvalue = value
        super().validate(castvalue, context=context)


class Enum(Validator):
    """
    requires one of a provided set of values
    eg. Enum(val1, val2, val3)
    """

    def __init__(self, *values):
        if not len(values):
            raise TypeError('Enum needs at least one value')

        self._values = set(values)
        self._valid_values = list(values)

    def validate(self, value, context=''):
        try:
            if value not in self._values:
                raise ValueError('{} is not in {}; {}'.format(
                    repr(value), repr(self._values), context))

        except TypeError as e:  # in case of unhashable (mutable) type
            e.args = e.args + ('error looking for {} in {}; {}'.format(
                repr(value), repr(self._values), context),)
            raise

    def __repr__(self):
        return '<Enum: {}>'.format(repr(self._values))


class OnOff(Validator):
    """
    requires either the string 'on' or 'off'
    """

    def __init__(self):
        self._validator = Enum('on', 'off')
        self._valid_values = self._validator._valid_values

    def validate(self, value, context=''):
        return self._validator.validate(value, context)


class Multiples(Ints):
    """
    A validator that checks if a value is an integer multiple of a
    fixed divisor. This class extends validators.Ints such that the
    value is also checked for being integer between an optional
    min_value and max_value. Furthermore this validator checks that
    the value is an integer multiple of an fixed, integer
    divisor. (i.e. value % divisor == 0)
    Args:
        divisor (integer), the value need the be a multiple of this divisor
    Inherited Args (see validators.Ints):
        max_value, value must be <= max_value
        min_value, value must be >= min_value
    """

    def __init__(self, divisor=1, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(divisor, int) or divisor <= 0:
            raise TypeError('divisor must be a positive integer')
        self._divisor = divisor
        self._valid_values = [divisor]

    def validate(self, value, context=''):
        super().validate(value=value, context=context)
        if not value % self._divisor == 0:
            raise ValueError('{} is not a multiple of {}; {}'.format(
                repr(value), repr(self._divisor), context))

    def __repr__(self):
        return super().__repr__()[:-1] + ', Multiples of {}>'.format(self._divisor)

    is_numeric = True


class PermissiveMultiples(Validator):
    """
    A validator that checks whether a value is an integer multiple
    of a fixed divisor (to within some precision). If both value and
    divisor are integers, the (exact) Multiples validator is used.

    We also allow negative values, meaning that zero by construction is
    always a valid value.

    Args:
        divisor: The number that the validated value should be an integer
            multiple of.
        precision: The maximally allowed absolute error between the value and
            the nearest true multiple
    """

    def __init__(self, divisor: Union[float, int, np.floating],
                 precision: float=1e-9) -> None:
        if divisor == 0:
            raise ValueError('Can not meaningfully check for multiples of'
                             ' zero.')
        self.divisor = divisor
        self.precision = precision
        self._numval = Numbers()
        if isinstance(divisor, int):
            self._mulval = Multiples(divisor=abs(divisor)) # type: Optional[Multiples]
        else:
            self._mulval = None
        self._valid_values = [divisor]

    def validate(self, value: Union[float, int, np.floating],
                 context: str='') -> None:
        """
        Validate the given value. Note that this validator does not use
        context for anything.
        """
        self._numval.validate(value)
        # if zero, it passes by definition
        if value == 0:
            return
        if self._mulval and isinstance(value, int):
            self._mulval.validate(abs(value))
        else:
            # floating-point division cannot be trusted, so we try to
            # multiply our way out of the problem by constructing true
            # multiples in the relevant range and see if `value` is one
            # of them (within rounding errors)
            divs = int(divmod(value, self.divisor)[0]) # type: ignore
            true_vals = np.array([n*self.divisor for n in range(divs, divs+2)])
            abs_errs = [abs(tv-value) for tv in true_vals]
            if min(abs_errs) > self.precision:
                raise ValueError('{} is not a multiple'.format(value) +
                                 ' of {}.'.format(self.divisor))

    def __repr__(self):
        repr = ('<PermissiveMultiples, Multiples of '
                '{} to within {}>'.format(self.divisor, self.precision))
        return repr

    is_numeric = True


class MultiType(Validator):
    """
    allow the union of several different validators
    for example to allow numbers as well as "off":
    MultiType(Numbers(), Enum("off"))
    The resulting validator acts as a logical OR between the
    different validators
    """

    def __init__(self, *validators):
        if not validators:
            raise TypeError('MultiType needs at least one Validator')

        for v in validators:
            if not isinstance(v, Validator):
                raise TypeError('each argument must be a Validator')

            if v.is_numeric:
                # if ANY of the contained validators is numeric,
                # the MultiType is considered numeric too.
                # this could cause problems if you want to sweep
                # from a non-numeric to a numeric value, so we
                # need to be careful about this in the sweep code
                self.is_numeric = True

        self._validators = tuple(validators)
        self._valid_values = []
        for val in self._validators:
            self._valid_values += val._valid_values
        self._valid_values = list(set(self._valid_values))

    def validate(self, value, context=''):
        args = ()
        for v in self._validators:
            try:
                v.validate(value, context)
                return
            except Exception as e:
                # collect the args from all validators so you can see why
                # each one failed
                args = args + e.args

        raise ValueError(*args)

    def __repr__(self):
        parts = (repr(v)[1:-1] for v in self._validators)
        return '<MultiType: {}>'.format(', '.join(parts))


class Arrays(Validator):
    """
    Validator for numerical numpy arrays
    Args:
        min_value:  Min value allowed, default inf.
        max_value: Max value allowed, default inf.
        shape: The shape of the array, standard numpy format.
    """

    validtypes = (int, float, np.integer, np.floating)

    def __init__(self, min_value: Union[float, int]=-float("inf"),
                 max_value: Union[float, int]=float("inf"),
                 shape: Tuple[int]=None) -> None:

        if isinstance(min_value, self.validtypes):
            self._min_value = min_value
        else:
            raise TypeError('min_value must be a number')

        valuesok = max_value > min_value  # type: ignore
        valuesok = cast(bool, valuesok)

        if isinstance(max_value, self.validtypes) and valuesok:
            self._max_value = max_value
        else:
            raise TypeError('max_value must be a number bigger than min_value')
        self._shape = shape

        if self._shape is None:
            self._valid_values = [np.array([min_value])]
        else:
            val_arr = np.empty(self._shape)
            val_arr.fill(min_value)
            self._valid_values = [val_arr]

    def validate(self, value, context=''):

        if not isinstance(value, np.ndarray):
            raise TypeError(
                '{} is not a numpy array; {}'.format(repr(value), context))

        if value.dtype not in self.validtypes:
            raise TypeError(
                '{} is not an int or float; {}'.format(repr(value), context))
        if self._shape is not None:
            if (np.shape(value) != self._shape):
                raise ValueError(
                    '{} does not have expected shape {}; {}'.format(
                        repr(value), self._shape, context))

        # Only check if max is not inf as it can be expensive for large arrays
        if self._max_value != (float("inf")):
            if not (np.max(value) <= self._max_value):
                raise ValueError(
                    '{} is invalid: all values must be between '
                    '{} and {} inclusive; {}'.format(
                        repr(value), self._min_value,
                        self._max_value, context))

        # Only check if min is not -inf as it can be expensive for large arrays
        if self._min_value != (-float("inf")):
            if not (self._min_value <= np.min(value)):
                raise ValueError(
                    '{} is invalid: all values must be between '
                    '{} and {} inclusive; {}'.format(
                        repr(value), self._min_value,
                        self._max_value, context))

    is_numeric = True

    def __repr__(self):
        minv = self._min_value if math.isfinite(self._min_value) else None
        maxv = self._max_value if math.isfinite(self._max_value) else None
        return '<Arrays{}, shape: {}>'.format(range_str(minv, maxv, 'v'),
                                              self._shape)


class Lists(Validator):
    """
    Validator for lists
    Args:
        elt_validator: used to validate the individual elements of the list
    """

    def __init__(self, elt_validator=Anything()):
        self._elt_validator = elt_validator
        self._valid_values = [elt_validator._valid_values]

    def __repr__(self):
        msg = '<Lists : '
        msg += self._elt_validator.__repr__() + '>'
        return msg

    def validate(self, value, context=''):
        if not isinstance(value, list):
            raise TypeError(
                '{} is not a list; {}'.format(repr(value), context))
        # Does not validate elements if not required to improve performance
        if not isinstance(self._elt_validator, Anything):
            for elt in value:
                self._elt_validator.validate(elt)


class Callable(Validator):
    """
    Validator for callables such as functions.
    """
    def __init__(self):
        self._valid_values = [lambda: 0]

    def validate(self, value, context=''):
        if not callable(value):
            raise TypeError(
                '{} is not a callable; {}'.format(repr(value), context))

    def __repr__(self):
        return '<Callable>'


class Dict(Validator):
    """
    Validator for dictionaries
    """

    def __init__(self, allowed_keys=None):
        """
        Validator for dictionary keys
        Args:
            allowed_keys (List): if set, all keys must be in allowed_keys
        """
        self.allowed_keys = allowed_keys
        self._valid_values = [{0: 1}]

    def validate(self, value, context=''):
        if not isinstance(value, dict):
            raise TypeError(
                '{} is not a dictionary; {}'.format(repr(value), context))

        if self.allowed_keys is not None:
            forbidden_keys = [key for key in value if key not in self.allowed_keys]
            if forbidden_keys:
                raise SyntaxError('Dictionary keys {} are not in allowed keys '
                                  '{}'.format(forbidden_keys,
                                              self.allowed_keys))


    def __repr__(self):
        if self.allowed_keys is None:
            return '<Dict>'
        else:
            return '<Dict {}>'.format(self.allowed_keys)
