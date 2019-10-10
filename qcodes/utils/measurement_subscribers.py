"""
This module contains example subscriber functions and classes that can be added
to a ``Measurement`` instance with ``add_subscriber``.
"""

import time
import datetime
from typing import (Optional, List, Tuple, Any, Union, MutableSequence,
                    MutableMapping)


class TextProgress:
    """
    A textual description of the progress of a measurement, printed in-place so
    that it doesn't clog up the notebook. Additionally, if
    ``total_measurements`` is set, it also prints the progress as a percentage,
    and predicts how long the measurement is going to take.
    ``total_measurements`` can be set after the measurement has been started,
    as the number of points might be determined only inside the measurement
    context.

    Args:
        total_measurements: The total number of data points that will be added
            to the dataset. Used to show the percentage of completion and
            extraploate the remaining time. Can also be set as an attribute
            after instantiating a TextProgress object.
        time_fmt: The format string used to print the finish time, as accepted
            by ``strftime``. Defaults to a format like "Fri Oct 11 12:34:56".
            If None, no date is printed. If total_measurements is not given,
            this parameter does nothing.

    Example::

        meas = qcodes.Measurement()

        meas.register_custom_parameter('x')
        meas.register_custom_parameter('y', setpoints=('x',))

        tp = TextProgress()
        meas.add_subscriber(tp, state=None)

        with meas.run() as datasaver:
            xs = np.linspace(-1, 1, 100)
            ys = np.linspace(0, 10, 20)

            tp.total_measurements = xs.size

            for x in xs:
                meas.add_result(('x', x), ('y', y))
                time.sleep(0.1)

        # Prints something like
        # 105 / 2000 (5.2%) - elapsed: 0:00:01 - remaining: 0:00:19 (Fri Oct 11 12:34:56)
    """

    def __init__(self, total_measurements: Optional[int] = None,
                 time_fmt: Optional[str] = "%a %b %d %H:%M:%S"):

        self.total_measurements = total_measurements
        self._start_time = None
        self._time_fmt = time_fmt

    def __call__(self, results_list: List[Tuple[Any, Any]], length: int,
                 state: Union[MutableSequence, MutableMapping]):

        if self._start_time is None:
            self._start_time = time.monotonic()

        msg = f"\r{length}"

        if self.total_measurements is not None:
            progress = length / self.total_measurements
            msg += f" / {self.total_measurements} ({progress * 100:.1f} %)"

        elapsed = time.monotonic() - self._start_time

        msg += f" - elapsed: {self._get_delta(elapsed)}"

        if self.total_measurements is not None and length > 1:
            remaining = elapsed / progress * (1 - progress)
            remaining_delta = self._get_delta(remaining)
            msg += f" - remaining: {remaining_delta}"

            if self._time_fmt is not None:
                finish_time = datetime.datetime.strftime(
                                  datetime.datetime.now() + remaining_delta,
                                  format=self._time_fmt)

                msg += f" ({finish_time})"

        print(msg, end='')

    def _get_delta(self, seconds: float) -> datetime.timedelta:
        return datetime.timedelta(seconds=int(seconds))
