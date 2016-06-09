import math
import numpy

BIGSTRING = 1000000000
BIGINT = int(1e18)


def validate_all(*args, context=''):
    '''
    takes a list of (validator, value) couplets and tests whether they are
    all valid, raising ValueError otherwise

    context: keyword-only arg with a string to include in the error message
        giving the user context for the error
    '''
    if context:
        context = '; ' + context

    for i, (validator, value) in enumerate(args):
        validator.validate(value, 'argument ' + str(i) + context)


def range_str(min_val, max_val, name):
    '''
    utility to represent ranges in Validator repr's
    '''
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
    '''
    base class for all value validators
    each should have its own constructor, and override:

    validate: function of two args: value, context
        value is what you're testing
        context is a string identifying the caller better

        raises an error (TypeError or ValueError) if the value fails

    is_numeric: is this a numeric type (so it can be swept)?
    '''
    def __init__(self):
        raise NotImplementedError

    def validate(self, value, context=''):
        raise NotImplementedError

    is_numeric = False  # is this a numeric type (so it can be swept)?


class Anything(Validator):
    '''allow any value to pass'''
    def __init__(self):
        pass

    def validate(self, value, context=''):
        pass

    is_numeric = True

    def __repr__(self):
        return '<Anything>'


class Bool(Validator):
    '''
    requires a boolean
    '''
    def __init__(self):
        pass

    def validate(self, value, context=''):
        if not isinstance(value, bool):
            raise TypeError(
                '{} is not Boolean; {}'.format(repr(value), context))

    def __repr__(self):
        return '<Boolean>'


class Strings(Validator):
    '''
    requires a string
    optional parameters min_length and max_length limit the allowed length
    to min_length <= len(value) <= max_length
    '''

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
        min_value (Optional[Union[float, int]):  Min value allowed, default inf
        max_value:  (Optional[Union[float, int]): Max  value allowed, default inf

    Raises:

    Todo:
        - fix raises
    """

    validtypes = (float, int, numpy.integer, numpy.floating)

    def __init__(self, min_value=-float("inf"), max_value=float("inf")):

        if isinstance(min_value, self.validtypes):
            self._min_value = min_value
        else:
            raise TypeError('min_value must be a number')

        if isinstance(max_value, self.validtypes) and max_value > min_value:
            self._max_value = max_value
        else:
            raise TypeError('max_value must be a number bigger than min_value')

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
    '''
    requires an integer
    optional parameters min_value and max_value enforce
    min_value <= value <= max_value
    '''

    validtypes = (int, numpy.integer)

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


class Enum(Validator):
    '''
    requires one of a provided set of values
    eg. Enum(val1, val2, val3)
    '''

    def __init__(self, *values):
        if not len(values):
            raise TypeError('Enum needs at least one value')

        self._values = set(values)

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


class MultiType(Validator):
    '''
    allow the union of several different validators
    for example to allow numbers as well as "off":
    MultiType(Numbers(), Enum("off"))
    '''

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
