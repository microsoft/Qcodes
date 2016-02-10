import math

BIGSTRING = 1000000000
BIGINT = int(1e18)


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


class Validator(object):
    '''
    base class for all value validators
    each should have its own constructor, and override is_valid and is_numeric
    '''
    def __init__(self):
        raise NotImplementedError

    def is_valid(self, value):
        raise NotImplementedError

    is_numeric = False  # is this a numeric type (so it can be swept)?


class Anything(Validator):
    '''allow any value to pass'''
    def __init__(self):
        pass

    def is_valid(self, value):
        return True

    is_numeric = True

    def __repr__(self):
        return '<Anything>'


class Bool(Validator):
    '''
    requires a boolean
    '''
    def __init__(self):
        pass

    def is_valid(self, value):
        return (isinstance(value, bool))

    def __repr_(self):
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

    def is_valid(self, value):
        return (isinstance(value, str) and
                self._min_length <= len(value) <= self._max_length)

    def __repr__(self):
        minv = self._min_length or None
        maxv = self._max_length if self._max_length < BIGSTRING else None
        return '<Strings{}>'.format(range_str(minv, maxv, 'len'))


class Numbers(Validator):
    '''
    requires a number, either int or float
    optional parameters min_value and max_value enforce
    min_value <= value <= max_value
    '''

    def __init__(self, min_value=-float("inf"), max_value=float("inf")):
        if isinstance(min_value, (float, int)):
            self._min_value = min_value
        else:
            raise TypeError('min_value must be a number')

        if isinstance(max_value, (float, int)) and max_value > min_value:
            self._max_value = max_value
        else:
            raise TypeError('max_value must be a number bigger than min_value')

    def is_valid(self, value):
        return (isinstance(value, (float, int)) and
                self._min_value <= value <= self._max_value)

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

    def __init__(self, min_value=-BIGINT, max_value=BIGINT):
        if isinstance(min_value, int):
            self._min_value = min_value
        else:
            raise TypeError('min_value must be an integer')

        if not isinstance(max_value, int):
            raise TypeError('max_value must be an integer')
        if max_value > min_value:
            self._max_value = max_value
        else:
            raise TypeError(
                'max_value must be an integer bigger than min_value')

    def is_valid(self, value):
        return (isinstance(value, int) and
                self._min_value <= value <= self._max_value)

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

    def is_valid(self, value):
        try:
            return value in self._values
        except TypeError:  # in case of unhashable (mutable) type
            return False

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

    def is_valid(self, value):
        for v in self._validators:
            if v.is_valid(value):
                return True

        return False

    def __repr__(self):
        parts = (repr(v)[1:-1] for v in self._validators)
        return '<MultiType: {}>'.format(', '.join(parts))
