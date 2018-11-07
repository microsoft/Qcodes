import builtins
import io
import json
import logging
import math
import numbers
import time

from collections import Iterator, Sequence, Mapping
from copy import deepcopy
from blinker import Signal

import numpy as np

_tprint_times = {}


class NumpyJSONEncoder(json.JSONEncoder):
    """Return numpy types as standard types."""
    # http://stackoverflow.com/questions/27050108/convert-numpy-type-to-python
    # http://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types/11389998#11389998

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif (isinstance(obj, numbers.Complex) and
              not isinstance(obj, numbers.Real)):
            return {
                '__dtype__': 'complex',
                're': float(obj.real),
                'im': float(obj.imag)
            }
        elif hasattr(obj, '_JSONEncoder'):
            # Use object's custom JSON encoder
            return obj._JSONEncoder()
        else:
            try:
                s = super(NumpyJSONEncoder, self).default(obj)
            except TypeError:
                # we cannot convert the object to JSON, just take a string
                s = str(obj)
            return s


def tprint(string, dt=1, tag='default'):
    """ Print progress of a loop every dt seconds """
    ptime = _tprint_times.get(tag, 0)
    if (time.time() - ptime) > dt:
        print(string)
        _tprint_times[tag] = time.time()


def is_sequence(obj):
    """
    Test if an object is a sequence.

    We do not consider strings or unordered collections like sets to be
    sequences, but we do accept iterators (such as generators)
    """
    return (isinstance(obj, (Iterator, Sequence, np.ndarray)) and
            not isinstance(obj, (str, bytes, io.IOBase)))


def is_sequence_of(obj, types=None, depth=None, shape=None):
    """
    Test if object is a sequence of entirely certain class(es).

    Args:
        obj (any): the object to test.

        types (Optional[Union[class, Tuple[class]]]): allowed type(s)
            if omitted, we just test the depth/shape

        depth (Optional[int]): level of nesting, ie if ``depth=2`` we expect
            a sequence of sequences. Default 1 unless ``shape`` is supplied.

        shape (Optional[Tuple[int]]): the shape of the sequence, ie its
            length in each dimension. If ``depth`` is omitted, but ``shape``
            included, we set ``depth = len(shape)``

    Returns:
        bool, True if every item in ``obj`` matches ``types``
    """
    if not is_sequence(obj):
        return False

    if shape in (None, ()):
        next_shape = None
        if depth is None:
            depth = 1
    else:
        if depth is None:
            depth = len(shape)
        elif depth != len(shape):
            raise ValueError('inconsistent depth and shape')

        if len(obj) != shape[0]:
            return False

        next_shape = shape[1:]

    for item in obj:
        if depth > 1:
            if not is_sequence_of(item, types, depth=depth - 1,
                                  shape=next_shape):
                return False
        elif types is not None and not isinstance(item, types):
            return False
    return True


def full_class(obj):
    """The full importable path to an object's class."""
    return type(obj).__module__ + '.' + type(obj).__name__


def named_repr(obj):
    """Enhance the standard repr() with the object's name attribute."""
    s = '<{}.{}: {} at {}>'.format(
        obj.__module__,
        type(obj).__name__,
        str(obj.name),
        id(obj))
    return s


def deep_update(dest, update):
    """
    Recursively update one JSON structure with another.

    Only dives into nested dicts; lists get replaced completely.
    If the original value is a dict and the new value is not, or vice versa,
    we also replace the value completely.
    """
    for k, v_update in update.items():
        v_dest = dest.get(k)
        if isinstance(v_update, Mapping) and isinstance(v_dest, Mapping):
            deep_update(v_dest, v_update)
        else:
            dest[k] = deepcopy(v_update)
    return dest


# could use numpy.arange here, but
# a) we don't want to require that as a dep so low level
# b) I'd like to be more flexible with the sign of step
def permissive_range(start, stop, step):
    """
    returns range (as a list of values) with floating point step

    inputs:
        start, stop, step

    always starts at start and moves toward stop,
    regardless of the sign of step
    """
    signed_step = abs(step) * (1 if stop > start else -1)
    # take off a tiny bit for rounding errors
    step_count = math.ceil((stop - start) / signed_step - 1e-10)
    return [start + i * signed_step for i in range(step_count)]


# This is very much related to the permissive_range but more
# strict on the input, start and endpoints are always included,
# and a sweep is only created if the step matches an integer
# number of points.
# numpy is a dependency anyways.
# Furthermore the sweep allows to take a number of points and generates
# an array with endpoints included, which is more intuitive to use in a sweep.
def make_sweep(start, stop, step=None, num=None):
    """
    Generate numbers over a specified interval.
    Requires `start` and `stop` and (`step` or `num`)
    The sign of `step` is not relevant.

    Args:
        start (Union[int, float]): The starting value of the sequence.
        stop (Union[int, float]): The end value of the sequence.
        step (Optional[Union[int, float]]):  Spacing between values.
        num (Optional[int]): Number of values to generate.

    Returns:
        numpy.linespace: numbers over a specified interval.

    Examples:
        >>> make_sweep(0, 10, num=5)
        [0.0, 2.5, 5.0, 7.5, 10.0]
        >>> make_sweep(5, 10, step=1)
        [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        >>> make_sweep(15, 10.5, step=1.5)
        >[15.0, 13.5, 12.0, 10.5]
    """
    if step and num:
        raise AttributeError('Don\'t use `step` and `num` at the same time.')
    if (step is None) and (num is None):
        raise ValueError('If you really want to go from `start` to '
                         '`stop` in one step, specify `num=2`.')
    if step is not None:
        steps = abs((stop - start) / step)
        tolerance = 1e-10
        steps_lo = int(np.floor(steps + tolerance))
        steps_hi = int(np.ceil(steps - tolerance))

        # if steps_lo != steps_hi:
            # raise ValueError(
            #     'Could not find an integer number of points for '
            #     'the the given `start`, `stop`, and `step` '
            #     'values. \nNumber of points is {:d} or {:d}.'
            #     .format(steps_lo + 1, steps_hi + 1))
        num = steps_hi + 1

    return np.linspace(start, stop, num=num).tolist()


def wait_secs(finish_clock):
    """
    calculate the number of seconds until a given clock time
    The clock time should be the result of time.perf_counter()
    Does NOT wait for this time.
    """
    delay = finish_clock - time.perf_counter()
    return max(delay, 0)


class LogCapture():

    """
    context manager to grab all log messages, optionally
    from a specific logger

    usage::

        with LogCapture() as logs:
            code_that_makes_logs(...)
        log_str = logs.value

    """

    def __init__(self, logger=logging.getLogger()):
        self.logger = logger

        self.stashed_handlers = self.logger.handlers[:]
        for handler in self.stashed_handlers:
            self.logger.removeHandler(handler)

    def __enter__(self):
        self.log_capture = io.StringIO()
        self.string_handler = logging.StreamHandler(self.log_capture)
        self.string_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.string_handler)
        return self

    def __exit__(self, type, value, tb):
        self.logger.removeHandler(self.string_handler)
        self.value = self.log_capture.getvalue()
        self.log_capture.close()

        for handler in self.stashed_handlers:
            self.logger.addHandler(handler)


def make_unique(s, existing):
    """
    make string s unique, able to be added to a sequence `existing` of
    existing names without duplication, by appending _<int> to it if needed
    """
    n = 1
    s_out = s
    existing = set(existing)

    while s_out in existing:
        n += 1
        s_out = '{}_{}'.format(s, n)

    return s_out


class DelegateAttributes:
    """
    Mixin class to create attributes of this object by
    delegating them to one or more dicts and/or objects

    Also fixes __dir__ so the delegated attributes will show up
    in dir() and autocomplete


    Attributes:
        delegate_attr_dicts (list): a list of names (strings) of dictionaries
            which are (or will be) attributes of self, whose keys should
            be treated as attributes of self
        delegate_attr_objects (list): a list of names (strings) of objects
            which are (or will be) attributes of self, whose attributes
            should be passed through to self
        omit_delegate_attrs (list): a list of attribute names (strings)
            to *not* delegate to any other dict or object

    any `None` entry is ignored

    attribute resolution order:
        1. real attributes of this object
        2. keys of each dict in delegate_attr_dicts (in order)
        3. attributes of each object in delegate_attr_objects (in order)
    """
    delegate_attr_dicts = []
    delegate_attr_objects = []
    omit_delegate_attrs = []

    def __getattr__(self, key):
        if key in self.omit_delegate_attrs:
            raise AttributeError("'{}' does not delegate attribute {}".format(
                self.__class__.__name__, key))

        for name in self.delegate_attr_dicts:
            if key == name:
                # needed to prevent infinite loops!
                raise AttributeError(
                    "dict '{}' has not been created in object '{}'".format(
                        key, self.__class__.__name__))
            try:
                d = getattr(self, name, None)
                if d is not None:
                    return d[key]
            except KeyError:
                pass

        for name in self.delegate_attr_objects:
            if key == name:
                raise AttributeError(
                    "object '{}' has not been created in object '{}'".format(
                        key, self.__class__.__name__))
            try:
                obj = getattr(self, name, None)
                if obj is not None:
                    return getattr(obj, key)
            except AttributeError:
                pass

        raise AttributeError(
            "'{}' object and its delegates have no attribute '{}'".format(
                self.__class__.__name__, key))

    def __dir__(self):
        names = super().__dir__()
        for name in self.delegate_attr_dicts:
            d = getattr(self, name, None)
            if d is not None:
                names += [k for k in d.keys()
                          if k not in self.omit_delegate_attrs]

        for name in self.delegate_attr_objects:
            obj = getattr(self, name, None)
            if obj is not None:
                names += [k for k in dir(obj)
                          if k not in self.omit_delegate_attrs]

        return sorted(set(names))


def strip_attrs(obj, whitelist=()):
    """
    Irreversibly remove all direct instance attributes of obj, to help with
    disposal, breaking circular references.

    Args:
        obj:  object to be stripped
        whitelist (list): list of names that are not stripped from the object
    """
    try:
        lst = set(list(obj.__dict__.keys())) - set(whitelist)
        for key in lst:
            try:
                del obj.__dict__[key]
            # TODO (giulioungaretti) fix bare-except
            except:
                pass
        # TODO (giulioungaretti) fix bare-except
    except:
        pass


def compare_dictionaries(dict_1, dict_2,
                         dict_1_name='d1',
                         dict_2_name='d2', path=""):
    """
    Compare two dictionaries recursively to find non matching elements

    Args:
        dict_1: dictionary 1
        dict_2: dictionary 2
        dict_1_name: optional name used in the differences string
        dict_2_name: ''
    Returns:
        dicts_equal:      Boolean
        dict_differences: formatted string containing the differences

    """
    err = ''
    key_err = ''
    value_err = ''
    old_path = path
    for k in dict_1.keys():
        path = old_path + "[%s]" % k
        if k not in dict_2.keys():
            key_err += "Key {}{} not in {}\n".format(
                dict_1_name, path, dict_2_name)
        else:
            if isinstance(dict_1[k], dict) and isinstance(dict_2[k], dict):
                err += compare_dictionaries(dict_1[k], dict_2[k],
                                            dict_1_name, dict_2_name, path)[1]
            else:
                match = (dict_1[k] == dict_2[k])

                # if values are equal-length numpy arrays, the result of
                # "==" is a bool array, so we need to 'all' it.
                # In any other case "==" returns a bool
                # TODO(alexcjohnson): actually, if *one* is a numpy array
                # and the other is another sequence with the same entries,
                # this will compare them as equal. Do we want this, or should
                # we require exact type match?
                if hasattr(match, 'all'):
                    match = match.all()

                if not match:
                    value_err += (
                        'Value of "{}{}" ("{}", type"{}") not same as\n'
                        '  "{}{}" ("{}", type"{}")\n\n').format(
                        dict_1_name, path, dict_1[k], type(dict_1[k]),
                        dict_2_name, path, dict_2[k], type(dict_2[k]))

    for k in dict_2.keys():
        path = old_path + "[{}]".format(k)
        if k not in dict_1.keys():
            key_err += "Key {}{} not in {}\n".format(
                dict_2_name, path, dict_1_name)

    dict_differences = key_err + value_err + err
    if len(dict_differences) == 0:
        dicts_equal = True
    else:
        dicts_equal = False
    return dicts_equal, dict_differences


def warn_units(class_name, instance):
    logging.warning('`units` is deprecated for the `' + class_name +
                    '` class, use `unit` instead. ' + repr(instance))


def get_last_input_cells(cells=3):
    """
    Get last input cell. Note that get_last_input_cell.globals must be set to
    the ipython globals
    Returns:
        last cell input if successful, else None
    """
    global In
    if 'In' in globals() or hasattr(builtins, 'In'):
        return In[-cells:]
    else:
        logging.warning('No input cells found')


def foreground_qt_window(window):
    """
    Try as hard as possible to bring a qt window to the front. This
    will use pywin32 if installed and running on windows as this
    seems to be the only reliable way to foreground a window. The
    build-in qt functions often doesn't work. Note that to use this
    with pyqtgraphs remote process you should use the ref in that module
    as in the example below.

    Args:
        window: handle to qt window to foreground
    Examples:
        >>> Qtplot.qt_helpers.foreground_qt_window(plot.win)
    """
    try:
        from win32gui import SetWindowPos
        import win32con
        # use the idea from
        # https://stackoverflow.com/questions/12118939/how-to-make-a-pyqt4-window-jump-to-the-front
        SetWindowPos(window.winId(),
                     win32con.HWND_TOPMOST, # = always on top. only reliable way to bring it to the front on windows
                     0, 0, 0, 0,
                     win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)
        SetWindowPos(window.winId(),
                     win32con.HWND_NOTOPMOST, # disable the always on top, but leave window at its top position
                     0, 0, 0, 0,
                     win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)
    except ImportError:
        pass
    window.show()
    window.raise_()
    window.activateWindow()


def smooth(y, window_size, order=3, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    Taken from: http://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
    Parameters

    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in
                range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


class SignalEmitter:
    """Class that allows other callables to connect to it listen for signals.

    Callables can be attached to a SignalEmitter via SignalEmitter.connect.
    If the SignalEmitter calls SignalEmitter.signal.send(*args, **kwargs), any
    callables are called with the respective args and kwargs.

    Args:
        initialize_signal: instantiate a blnker.Signal object. If set to False,
            self.Signal needs to be set later on. This could be useful when you
            want a single Signal that is shared by all class instances.
        multiple_senders: Allow to be connected to multiple senders.
            If False, when connected to a second SignalEmitter, the connection
            to the previous SignalEmitter is disconnected

    Note:
        The SignalEmitter has protection against infinite recursions resulting
        from signal emitters calling each other. This is done by keeping track
        of the signal chain. However, it does not protect against infinite
        recursions from signals sent from objects that are not signal emitters.
    """
    # Signal used for connecting to parameter via SignalEmitter.connect method
    signal = None

    def __init__(self, initialize_signal: bool=True,
                 multiple_senders: bool = True):
        self._signal_chain = []
        if initialize_signal:
            self.signal = Signal()

        self._signal_modifiers = {
            'offset': None,
            'scale': None
        }

        # By default self is not connected to any other SignalEmitter
        self.sender = None

        self.multiple_senders = multiple_senders

    def connect(self, receiver, update=False, offset: float = None,
                scale: float = None):
        """Connect a receiver, which can be another SignalEmitter.

        If a SignalEmitter is passed, the __call__ method is invoked.

        Args:
            receiver: Receiver to be connected to this SignalEmitter's signal.
            offset: Optional offset to apply to emitted value
            scale: Optional scale to apply to emitted value

        Note:
            If offset or scale is provided, the emitted value should be a number.
            If an emitted signal contains 'value' as kwarg, this will be
            modified. Otherwise the first arg (sender) will be modified.
        """
        if self.signal is None:
            self.signal = Signal()

        if isinstance(receiver, SignalEmitter):
            # Remove any previous sender if multiple_senders is False
            if not receiver.multiple_senders and receiver.sender is not None:
                receiver.sender.disconnect(receiver)

            receiver._signal_modifiers['offset'] = offset
            receiver._signal_modifiers['scale'] = scale
            self.signal.connect(receiver._signal_call)
            receiver.sender = self

        else:
            self.signal.connect(receiver)

        if update:
            # Update callable with current value
            value = self()
            if scale is not None:
                if callable(scale):
                    scale = scale(self)
                value *= scale
            if offset is not None:
                if callable(offset):
                    offset = offset(self)
                value += offset

            receiver(value)

    def disconnect(self, callable):
        """disconnect a callable from a SignalEmitter.

        Note:
            Does not raise error if callable is not connected in the first place
            """
        if isinstance(callable, SignalEmitter):
            callable = callable._signal_call

        for receiver_ref in list(self.signal.receivers.values()):
            receiver = receiver_ref()
            if receiver == callable:
                self.signal.disconnect(callable)

    def _signal_call(self, sender, *args, **kwargs):
        """Method that is called instead of standard __call__ for SignalEmitters

        This method ensures that the actual __call__ is only invoked if this has
        not previously been done during the signal chain.
        """
        if self not in self._signal_chain:
            value = kwargs.get('value', sender)

            # If any modifier is set,
            if self._signal_modifiers['scale'] is not None:
                scale = self._signal_modifiers['scale']
                if callable(scale):
                    scale = scale(self.sender)
                value *= scale

            if self._signal_modifiers['offset'] is not None:
                offset = self._signal_modifiers['offset']
                if callable(offset):
                    offset = offset(self.sender)
                value += offset

            if 'value' in kwargs:
                kwargs['value'] = value
            else:
                sender = value
            return self(sender, *args, signal_chain=self._signal_chain, **kwargs)