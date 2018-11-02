import math
from typing import Union, Optional, Tuple, Any, Hashable
# rename on import since this file implements its own classes
# with these names.
from typing import Callable as TCallable
from typing import Sequence as TSequence
from typing import List as TList

import collections

import numpy as np

BIGSTRING = 1000000000
BIGINT = int(1e18)

numbertypes = Union[float, int, np.floating, np.integer]


def validate_all(*args, context: str='') -> None:
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


def range_str(min_val: Union[float, int], max_val: Union[float, int],
              name: str) -> str:
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
    each validator should implement:

    __init__: here a private attribute, _valid_values, should be set.
        _valid_values must be a tuple of at least one valid value.
        If possible, it should include all valid values. The purpose of
        this attribute is to make it possible to find a valid value for
        a Parameter, given its validator.

    validate: function of two args: value, context
        value is what you're testing
        context is a string identifying the caller better

        raises an error (TypeError or ValueError) if the value fails

    is_numeric: A boolean flag that marks if this a numeric type.

    The base class implements:

    valid_values: a property exposing _valid_values, which is a tuple
        of examples of valid values. For very simple validators, like
        Bool or Enum, the tuple contains all valid values, but in general
        it just holds SOME valid values. These example values are intended
        to be useful when simulating instruments.

    Alternatively you may override valid_values and provide your own
    implementation of getting valid values.
    """
    _valid_values: Tuple = ()
    is_numeric = False  # is this a numeric type (so it can be swept)?

    def validate(self, value, context: str=''):
        raise NotImplementedError

    @property
    def valid_values(self) -> Tuple:
        return self._valid_values


class Anything(Validator):
    """allow any value to pass"""

    def __init__(self) -> None:
        self._valid_values = (0,)

    def validate(self, value: Any, context: str=''):
        pass
    # NOTE(giulioungaretti): why is_numeric?
    # it allows for set_step in parameter
    # TODO(giulioungaretti): possible refactor
    is_numeric = True

    def __repr__(self) -> str:
        return '<Anything>'


class Nothing(Validator):
    """allow no value to pass"""

    def __init__(self, reason: str) -> None:
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

    def __init__(self) -> None:
        self._valid_values = (True, False)

    def validate(self, value: bool, context: str='') -> None:
        if not isinstance(value, bool) and not isinstance(value, np.bool8):
            raise TypeError(
                '{} is not Boolean; {}'.format(repr(value), context))

    def __repr__(self) -> str:
        return '<Boolean>'


class Strings(Validator):
    """
    requires a string
    optional parameters min_length and max_length limit the allowed length
    to min_length <= len(value) <= max_length
    """

    def __init__(self, min_length: int=0, max_length: int=BIGSTRING) -> None:
        if isinstance(min_length, int) and min_length >= 0:
            self._min_length = min_length
        else:
            raise TypeError('min_length must be a non-negative integer')
        if isinstance(max_length, int) and max_length >= max(min_length, 1):
            self._max_length = max_length
        else:
            raise TypeError('max_length must be a positive integer '
                            'no smaller than min_length')
        self._valid_values = ('.'*min_length,)

    def validate(self, value: str, context: str=''):
        if not isinstance(value, str):
            raise TypeError(
                '{} is not a string; {}'.format(repr(value), context))

        vallen = len(value)
        if vallen < self._min_length or vallen > self._max_length:
            raise ValueError(
                '{} is invalid: length must be between '
                '{} and {} inclusive; {}'.format(
                    repr(value), self._min_length, self._max_length, context))

    def __repr__(self) -> str:
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

    def __init__(self, min_value: numbertypes=-float("inf"),
                 max_value: numbertypes=float("inf")) -> None:

        if isinstance(min_value, self.validtypes):
            self._min_value = min_value
        else:
            raise TypeError('min_value must be a number')

        valuesok = max_value > min_value

        if isinstance(max_value, self.validtypes) and valuesok:
            self._max_value = max_value
        else:
            raise TypeError('max_value must be a number bigger than min_value')

        self._valid_values = (min_value, max_value)

    def validate(self, value: numbertypes, context: str='') -> None:
        if not isinstance(value, self.validtypes):
            raise TypeError(
                '{} is not an int or float; {}'.format(repr(value), context))

        if not (self._min_value <= value <= self._max_value):
            raise ValueError(
                '{} is invalid: must be between '
                '{} and {} inclusive; {}'.format(
                    repr(value), self._min_value, self._max_value, context))

    is_numeric = True

    def __repr__(self) -> str:
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
    inttypes = Union[int, np.integer]

    def __init__(self, min_value: inttypes=-BIGINT,
                 max_value: inttypes=BIGINT) -> None:
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

        self._valid_values = (min_value, max_value)

    def validate(self, value: inttypes, context: str='') -> None:
        if not isinstance(value, self.validtypes):
            raise TypeError(
                '{} is not an int; {}'.format(repr(value), context))

        if not (self._min_value <= value <= self._max_value):
            raise ValueError(
                '{} is invalid: must be between '
                '{} and {} inclusive; {}'.format(
                    repr(value), self._min_value, self._max_value, context))

    is_numeric = True

    def __repr__(self) -> str:
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

    def validate(self, value: numbertypes, context: str='') -> None:
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

    def __init__(self, *values: Hashable) -> None:
        if not len(values):
            raise TypeError('Enum needs at least one value')

        self._values = set(values)
        self._valid_values = tuple(values)

    def validate(self, value, context: str='') -> None:
        try:
            if value not in self._values:
                raise ValueError('{} is not in {}; {}'.format(
                    repr(value), repr(self._values), context))

        except TypeError as e:  # in case of unhashable (mutable) type
            e.args = e.args + ('error looking for {} in {}; {}'.format(
                repr(value), repr(self._values), context),)
            raise

    def __repr__(self) -> str:
        return '<Enum: {}>'.format(repr(self._values))


class OnOff(Validator):
    """
    requires either the string 'on' or 'off'
    """

    def __init__(self) -> None:
        self._validator = Enum('on', 'off')
        self._valid_values = self._validator._valid_values

    def validate(self, value: str, context: str='') -> None:
        self._validator.validate(value, context)


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

    def __init__(self, divisor: int=1, **kwargs) -> None:
        super().__init__(**kwargs)
        if not isinstance(divisor, int) or divisor <= 0:
            raise TypeError('divisor must be a positive integer')
        self._divisor = divisor
        self._valid_values = (divisor,)

    def validate(self, value: int, context: str='') -> None:
        super().validate(value=value, context=context)
        if not value % self._divisor == 0:
            raise ValueError('{} is not a multiple of {}; {}'.format(
                repr(value), repr(self._divisor), context))

    def __repr__(self) -> str:
        return super().__repr__()[:-1] + f', Multiples of {self._divisor}>'

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


    def __init__(self, divisor: numbertypes,
                 precision: float=1e-9) -> None:
        if divisor == 0:
            raise ValueError('Can not meaningfully check for multiples of'
                             ' zero.')
        self.divisor = divisor
        self.precision = precision
        self._numval = Numbers()
        if isinstance(divisor, int):
            self._mulval: Optional[Multiples] = Multiples(divisor=abs(divisor))
        else:
            self._mulval = None
        self._valid_values = (divisor,)

    def validate(self, value: numbertypes,
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
            divs = int(divmod(value, self.divisor)[0])
            true_vals = np.array([n*self.divisor for n in range(divs, divs+2)])
            abs_errs = [abs(tv-value) for tv in true_vals]
            if min(abs_errs) > self.precision:
                raise ValueError('{} is not a multiple'.format(value) +
                                 ' of {}.'.format(self.divisor))

    def __repr__(self) -> str:
        repr_str = ('<PermissiveMultiples, Multiples of '
                    '{} to within {}>'.format(self.divisor, self.precision))
        return repr_str

    is_numeric = True


class MultiType(Validator):
    """
    allow the union of several different validators
    for example to allow numbers as well as "off":
    MultiType(Numbers(), Enum("off"))
    The resulting validator acts as a logical OR between the
    different validators
    """

    def __init__(self, *validators: Validator) -> None:
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
        self._valid_values = tuple(vval for v in self._validators
                                   for vval in v._valid_values)

    def validate(self, value: Any, context: str='') -> None:
        args: TList[str] = []
        for v in self._validators:
            try:
                v.validate(value, context)
                return
            except Exception as e:
                # collect the args from all validators so you can see why
                # each one failed
                args = args + list(e.args)

        raise ValueError(*args)

    def __repr__(self) -> str:
        parts = (repr(v)[1:-1] for v in self._validators)
        return '<MultiType: {}>'.format(', '.join(parts))


class Arrays(Validator):
    """
    Validator for numerical numpy arrays

    Args:
        min_value:  Min value allowed, default inf.
        max_value: Max value allowed, default inf.
        shape: The shape of the array, tuple of either ints or Callables taking
            no arguments that return the size along that dim as an int.
    """

    validtypes = (int, float, np.integer, np.floating)

    def __init__(self, min_value: numbertypes=-float("inf"),
                 max_value: numbertypes=float("inf"),
                 shape: TSequence[Union[int, TCallable[[], int]]]=None) -> None:

        if isinstance(min_value, self.validtypes):
            self._min_value = min_value
        else:
            raise TypeError('min_value must be a number')

        valuesok = max_value > min_value

        if isinstance(max_value, self.validtypes) and valuesok:
            self._max_value = max_value
        else:
            raise TypeError('max_value must be a number bigger than min_value')
        self._shape = shape

    @property
    def valid_values(self) -> Tuple[np.ndarray]:
        shape = self.shape
        if shape is None:
            return (np.array([self._min_value]),)
        else:
            val_arr = np.empty(shape)
            val_arr.fill(self._min_value)
            return (val_arr,)

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        if self._shape is None:
            return None
        shape_array = []
        for s in self._shape:
            if callable(s):
                shape_array.append(s())
            else:
                shape_array.append(s)
        shape = tuple(shape_array)
        return shape

    def validate(self, value: np.ndarray, context: str='') -> None:

        if not isinstance(value, np.ndarray):
            raise TypeError(
                '{} is not a numpy array; {}'.format(repr(value), context))

        if value.dtype not in self.validtypes:
            raise TypeError(
                '{} is not an int or float; {}'.format(repr(value), context))
        if self.shape is not None:
            shape = self.shape
            if np.shape(value) != shape:
                raise ValueError(
                    f'{repr(value)} does not have expected shape {shape},'
                    f'it has shape {np.shape(value)}; {context}')

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

    def __repr__(self) -> str:
        minv = self._min_value if math.isfinite(self._min_value) else None
        maxv = self._max_value if math.isfinite(self._max_value) else None
        return '<Arrays{}, shape: {}>'.format(range_str(minv, maxv, 'v'),
                                              self.shape)


class Lists(Validator):
    """
    Validator for lists
    Args:
        elt_validator: used to validate the individual elements of the list
    """

    def __init__(self, elt_validator: Validator=Anything()) -> None:
        self._elt_validator = elt_validator
        self._valid_values = ([vval for vval in elt_validator._valid_values],)

    def __repr__(self) -> str:
        msg = '<Lists : '
        msg += self._elt_validator.__repr__() + '>'
        return msg

    def validate(self, value: TList[Anything], context: str='') -> None:
        if not isinstance(value, list):
            raise TypeError(
                '{} is not a list; {}'.format(repr(value), context))
        # Does not validate elements if not required to improve performance
        if not isinstance(self._elt_validator, Anything):
            for elt in value:
                self._elt_validator.validate(elt)


class Sequence(Validator):
    """
    Validator for Sequences
    Args:
        elt_validator: used to validate the individual elements of the Sequence
    """

    def __init__(self, elt_validator: Validator=Anything(),
                 length: int=None, require_sorted: bool=False) -> None:
        self._elt_validator = elt_validator
        self._length = length
        self._require_sorted = require_sorted
        self._valid_values = ([vval for vval in elt_validator._valid_values],)

    def __repr__(self) -> str:
        msg = '<Sequence : '
        msg += f'len: {self._length} '
        msg += f'sorted: {self._require_sorted} '
        msg += self._elt_validator.__repr__() + '>'
        return msg

    def validate(self, value: collections.abc.Sequence, context: str='') -> None:
        if not isinstance(value, collections.abc.Sequence):
            raise TypeError(
                '{} is not a sequence; {}'.format(repr(value), context))
        if self._length and not len(value) == self._length:
            raise ValueError(
                f'{repr(value)} has not length {self._length} but {len(value)}')
        if self._require_sorted and sorted(value) != value:
            raise ValueError(
                f'{repr(value)} is required to be sorted.')
        # Does not validate elements if not required to improve performance
        if not isinstance(self._elt_validator, Anything):
            for elt in value:
                self._elt_validator.validate(elt)


class Callable(Validator):
    """
    Validator for callables such as functions.
    """
    def __init__(self) -> None:
        self._valid_values = (lambda: 0,)

    def validate(self, value: TCallable, context: str='') -> None:
        if not callable(value):
            raise TypeError(
                '{} is not a callable; {}'.format(repr(value), context))

    def __repr__(self) -> str:
        return '<Callable>'


class Dict(Validator):
    """
    Validator for dictionaries
    """

    def __init__(self, allowed_keys: TSequence[Hashable]=None) -> None:
        """
        Validator for dictionary keys
        Args:
            allowed_keys (List): if set, all keys must be in allowed_keys
        """
        self.allowed_keys = allowed_keys
        self._valid_values =({0: 1},)

    def validate(self, value: dict, context: str='') -> None:
        if not isinstance(value, dict):
            raise TypeError(
                '{} is not a dictionary; {}'.format(repr(value), context))

        if self.allowed_keys is not None:
            forbidden_keys = [key for key in value if key not in self.allowed_keys]
            if forbidden_keys:
                raise SyntaxError('Dictionary keys {} are not in allowed keys '
                                  '{}'.format(forbidden_keys,
                                              self.allowed_keys))

    def __repr__(self) -> str:
        if self.allowed_keys is None:
            return '<Dict>'
        else:
            return '<Dict {}>'.format(self.allowed_keys)
