import warnings
from typing import TYPE_CHECKING, Any, List

import numpy

from qcodes.instrument.parameter import ParameterWithSetpoints
from .message_builder import MessageBuilder
from . import constants
from .KeysightB1500_module import fmt_response_base_parser, _FMTResponse, \
    MeasurementNotTaken, convert_dummy_val_to_nan

if TYPE_CHECKING:
    from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1517A \
        import B1517A
    from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1500_base \
        import KeysightB1500


class SamplingMeasurement(ParameterWithSetpoints):
    """
    Performs sampling measurement using semiconductor
    parameter analyzer B1500A.
    """

    _timeout_response_factor = 10.00
    # This factor is a bit higher than the ratio between
    # the measured measurement-time and the calculated measurement
    # (from the user input). Check :get_raw: method to find its usage.

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, **kwargs)

        self.instrument: "B1517A"
        self.root_instrument: "KeysightB1500"

        self.data = _FMTResponse(None, None, None, None)

    def get_raw(self) -> numpy.ndarray:
        """
        This performs sampling  measurements. However since the measurement
        time can vary from few seconds to hundreds of minutes we first set
        the visa time out.
        The visa time-out should be longer than the time it takes to finish
        the sampling measurement. The reason is that while measurement is
        running the data keeps on appending in the buffer of SPA. Only when
        the measurement is finished the data is returned to the VISA handle.
        Hence during this time the VISA is idle and waiting for the response.
        If the timeout is lower than the total run time of the measurement,
        VISA will give error.
        We set the Visa timeout to be the `measurement_time` times the
        `_timeout_response_factor`. Strictly speaking the timeout should be
        just longer than the measurement time.
        """

        measurement_time = self.instrument._total_measurement_time()
        time_out = measurement_time * self._timeout_response_factor
        default_timeout = self.root_instrument.timeout()

        # if time out to be set is lower than the default value
        # then keep default
        if time_out < default_timeout:
            time_out = default_timeout

        self.root_instrument.write(MessageBuilder().fmt(1, 0).message)

        with self.root_instrument.timeout.set_to(time_out):
            raw_data = self.root_instrument.ask(
                MessageBuilder().xe().message)

        self.data = fmt_response_base_parser(raw_data)
        convert_dummy_val_to_nan(self.data)
        return numpy.array(self.data.value)

    def compliance(self) -> List[int]:
        """
        check for the status other than "N" (normal) and output the
        number of data values which were not measured under "N" (normal)
        status.

        For the list of all the status values and their meaning refer to
        :class:`.constants.MeasurementStatus`.

        """

        if self.data.status is None:
            raise MeasurementNotTaken('First run sampling_measurement'
                                      ' method to generate the data')
        else:
            data = self.data
            total_count = len(data.status)
            normal_count = data.status.count(constants.MeasurementStatus.N.name)
            exception_count = total_count - normal_count
            if total_count == normal_count:
                print('All measurements are normal')
            else:
                indices = [i for i, x in enumerate(data.status)
                           if x == "C" or x == "T"]
                warnings.warn(f'{str(exception_count)} measurements were '
                              f'out of compliance at {str(indices)}')

            compliance_list = [constants.MeasurementError[key].value
                               for key in data.status]
            return compliance_list
