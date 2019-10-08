"""
This module contains example subscriber functions and classes that can be added
to a ``Measurement`` instance with ``add_subscriber``.
"""

import time
import datetime
from typing import Optional, Tuple, Union, MutableSequence, MutableMapping

class TextProgress:
    """
    A textual description of the progress of a measurement, printed in-place so
    that it doesn't clog up the notebook. Additionally, if the attribute
    ``total_measurements`` is set, it also prints the progress as a percentage,
    and attempts to predict how long until the measurement is finished.
    ``total_measurements`` is an attribute so that it can be added after the
    measurement has been started, as the number of points might be determined
    only inside the ``Runner`` context.

    Example::

        meas = qcodes.Measurement()

        meas.register_custom_parameter('x')
        meas.register_custom_parameter('y', setpoints=('x',))

        tp = TextProgress()
        meas.add_subscriber(tp, state=None)

        with meas.run() as datasaver:
            xs = np.linspace(-1, 1, 100)
            ys = np.linspace(0, 10, 20)

            tp.total_measurements = xs.size * ys.size

            for x in xs:
                for y in ys:
                    meas.add_result(('x', x), ('y', y))
                    time.sleep(0.1)

        # Prints something like
        # 105 / 2000 (5.2%) - elapsed: 0:00:01 - remaining: 0:00:19
    """

    def __init__(self):

        self.total_measurements: Optional[int] = None
        self._start_time = None

    def __call__(self, results_list: Tuple, length: int,
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
            remaining = (1 - progress) / progress * elapsed
            msg += f" - remaining: {self._get_delta(remaining)}"

        print(msg, end='')

    def _get_delta(self, seconds: float) -> str:
        return datetime.timedelta(seconds=int(seconds))
