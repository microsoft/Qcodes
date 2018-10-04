import numpy as np
import logging
from typing import cast, Union
import time

from qcodes import InstrumentChannel, Instrument
from qcodes.utils.validators import Numbers, Enum, MultiType

from qcodes.instrument_drivers.Keysight.N52xx.trace import N52xxTrace


logger = logging.getLogger()


class N52xxChannel(InstrumentChannel):
    """
    Allows operations on specific channels.
    """
    def __init__(
            self, parent: 'Instrument', channel: int,
            measurement_type: str="S-parameter"
    ) ->None:

        super().__init__(parent, f"channel{channel}")

        self._channel = channel
        self._measurement_type = measurement_type

        present_channels = parent.list_existing_channel_numbers()
        self._present_on_instrument = channel in present_channels

        self.add_parameter(
            'power',
            label='Power',
            get_cmd=f'SOUR{self._channel}:POW?',
            get_parser=float,
            set_cmd=f'SOUR{self._channel}:POW {{:.2f}}',
            unit='dBm',
            vals=Numbers(
                min_value=self.parent.min_power,
                max_value=self.parent.max_power
            ),
            set_parser=float
        )

        self.add_parameter(
            'if_bandwidth',
            label='IF Bandwidth',
            get_cmd=f'SENS{self._channel}:BAND?',
            get_parser=float,
            set_cmd=f'SENS{self._channel}:BAND {{:.2f}}',
            unit='Hz',
            vals=Numbers(min_value=1, max_value=15e6)
        )

        self.add_parameter(
            'averages_enabled',
            label='Averages Enabled',
            get_cmd=f"SENS{self._channel}:AVER?",
            set_cmd=f"SENS{self._channel}:AVER {{}}",
            val_mapping={True: '1', False: '0'}
        )

        self.add_parameter(
            'averages',
            label='Averages',
            get_cmd=f'SENS{self._channel}:AVER:COUN?',
            get_parser=int,
            set_cmd=f'SENS{self._channel}:AVER:COUN {{:d}}',
            unit='',
            vals=Numbers(min_value=1, max_value=65536)
        )

        # Setting frequency range
        self.add_parameter(
            'start',
            label='Start Frequency',
            get_cmd=f'SENS{self._channel}:FREQ:STAR?',
            get_parser=float,
            set_cmd=f'SENS{self._channel}:FREQ:STAR {{}}',
            unit='',
            vals=Numbers(
                min_value=self.parent.min_freq,
                max_value=self.parent.max_freq
            )
        )

        self.add_parameter(
            'stop',
            label='Stop Frequency',
            get_cmd=f'SENS{self._channel}:FREQ:STOP?',
            get_parser=float,
            set_cmd=f'SENS{self._channel}:FREQ:STOP {{}}',
            unit='',
            vals=Numbers(
                min_value=self.parent.min_freq,
                max_value=self.parent.max_freq
            )
        )

        self.add_parameter(
            'points',
            label='Frequency Points',
            get_cmd=f'SENS{self._channel}:SWE:POIN?',
            get_parser=int,
            set_cmd=f'SENS{self._channel}:SWE:POIN {{}}',
            unit='',
            vals=Numbers(min_value=1, max_value=100001)
        )

        self.add_parameter(
            'electrical_delay',
            label='Electrical Delay',
            get_cmd=f'CALC{self._channel}:CORR:EDEL:TIME?',
            get_parser=float,
            set_cmd=f'CALC{self._channel}:CORR:EDEL:TIME {{:.6e}}',
            unit='s',
            vals=Numbers(min_value=0, max_value=100000)
        )

        self.add_parameter(
            'sweep_time',
            label='Time',
            get_cmd=f'SENS{self._channel}:SWE:TIME?',
            get_parser=float,
            # Make sure we are in stepped sweep
            set_cmd=f'SENS{self._channel}:SWE:TIME {{:.6e}}',
            unit='s',
            vals=Numbers(0, 1e6)
        )

        self.add_parameter(
            'dwell_time',
            label='Time',
            get_cmd=f'SENS{self._channel}:SWE:DWEL?',
            get_parser=float,
            # Make sure we are in stepped sweep
            set_cmd=f'SENS{self._channel}:SWE:GEN STEP;'
                    f'SENS{self._channel}:SWE:DWEL {{:.6e}}',
            unit='s',
            vals=MultiType(Numbers(0, 1e6), Enum("min", "max"))
        )

        self.add_parameter(
            'sweep_mode',
            label='Mode',
            get_cmd=f'SENS{self._channel}:SWE:MODE?',
            set_cmd=f'SENS{self._channel}:SWE:MODE {{}}',
            vals=Enum("HOLD", "CONT", "GRO", "SING")
        )

        self.add_parameter(
            "sensor_correction",
            get_cmd=f"SENS{self._channel}:CORR?",
            set_cmd=f"SENS{self._channel}:CORR {{}}",
            val_mapping={True: '1', False: '0'}
        )

        self._traces = self._load_traces_from_instrument()

    def _load_traces_from_instrument(self) ->list:
        """
        Interface to access traces on the instrument

        Returns:
            dict: keys are trace names, values are instance of `N52xxTrace`
        """
        if not self._present_on_instrument:
            return []

        result = self.ask(f"CALC{self._channel}:PAR:CAT:EXT?")
        if result == "NO CATALOG":
            return []

        trace_info = result.strip("\"").split(",")
        trace_names = trace_info[::2]
        trace_types = trace_info[1::2]

        parent = cast(Instrument, self.parent)

        # TODO: The type of trace returned should depend on
        # TODO: self._measurement_type
        return [N52xxTrace(
            parent, self, name, trace_type, present_on_instrument=True)
            for name, trace_type in zip(trace_names, trace_types)
        ]

    @property
    def trace(self) ->list:
        """
        List all traces on the instrument
        """
        return [trace for trace in self._traces if trace.present_on_instrument]

    @property
    def channel_number(self):
        return self._channel

    def add_trace(self, tr_type: str) -> 'N52xxTrace':
        """
        Add a trace the instrument.

        Args:
            tr_type (str): Currently only S-parameter types are supported, which
                have the format `Sxy` where x and y are integers.

        Returns:
            trace (N52xxTrace)
        """
        name = f"CH{self._channel}_{tr_type}"

        trace = {tr.short_name: tr for tr in self.trace}.get(name, None)
        if trace is not None:
            return trace

        parent = cast(Instrument, self.parent)
        # TODO: The type of trace returned should depend on
        # TODO: self._measurement_type
        trace = N52xxTrace(parent, self, name, tr_type)
        self._traces.append(trace)
        trace.upload_to_instrument()
        trace.select()
        return trace

    def delete_trace(self, index: Union[int, str]) ->None:
        """
        Deletes the trace on the instrument

        Args:
            index (str or int): Either the name of the trace to delete or "all"
        """
        if index == "all":
            for trace in self.trace:
                trace.delete()
        else:
            self.trace[index].delete()

    @property
    def present_on_instrument(self):
        return self._present_on_instrument

    def upload_channel_to_instrument(self):

        if self._measurement_type == "S-parameter":
            type_string = "S11"
        else:
            raise ValueError("currently, only S-parameter measurements are "
                             "supported")

        existing_meas = self.parent.list_existing_measurement_numbers()
        new_meas = 1
        while new_meas in existing_meas:
            new_meas += 1

        # Defining a new measurement on an, as yet, non-existing channel
        # will create that channel
        self.parent.write(
            f"CALC{self._channel}:MEAS{new_meas}:DEF '{type_string}'")

        self._present_on_instrument = True

    def select(self) ->None:
        """
        A channel must be selected (active) to modify its settings. A channel
        is selected by selecting a trace on that channel
        """
        if len(self.trace) == 0:
            raise UserWarning("Cannot select channel as no traces have been "
                              "defined ")
        else:
            self.trace[0].select()

    def delete(self):
        self.write(f"SYST:CHAN:DEL {self._channel}")
        self._present_on_instrument = False

    def _assert_instrument_presence(self):
        if not self._present_on_instrument:
            raise RuntimeError("The channel is not present (anymore) on the "
                               "instrument. It was either deleted or never "
                               "uploaded in the first place")

    def write(self, cmd: str) -> None:
        self._assert_instrument_presence()
        super().write(cmd)

    def ask(self, cmd: str) -> str:
        self._assert_instrument_presence()
        return super().ask(cmd)

    def run_sweep(self, averages: int =1, blocking: bool=True) ->None:
        """
        Run a sweep

        Args:
            averages (int): The number of averages
            blocking (bool): If True, this method will block until the sweep
                                has finished
        """
        self.select()

        if averages == 1:
            self.averages_enabled(False)
            self.sweep_mode('SING')
        else:
            self.averages_enabled(True)
            self.averages(averages)

            self.write(f'SENS{self._channel}:AVER:CLE')
            self.write(f'SENS{self._channel}:SWE:GRO:COUN {averages}')
            self.sweep_mode('GRO')

        if blocking:
            self.block_while_not_hold()

    def block_while_not_hold(self) ->None:
        """
        Block until a sweep has finished
        """
        try:
            # Once the sweep mode is in hold, we know we're done
            # Note that if no triggers are received, we can get stuck in an
            # infinite loop
            while self.sweep_mode() != 'HOLD':
                time.sleep(0.1)
        except KeyboardInterrupt:
            # If the user aborts because (s)he is stuck in the infinite loop
            # mentioned above, provide a hint of what can be wrong.
            msg = "User abort detected. "
            source = self.parent.trigger_source()

            if source == "MAN":
                msg += "The trigger source is manual. Are you sure this is " \
                       "correct? Please set the correct source with the " \
                       "'trigger_source' parameter"
            elif source == "EXT":
                msg += "The trigger source is external. Is the trigger " \
                       "source functional?"

            logger.warning(msg)

    def get_snp_data(self, ports: list=None) ->np.ndarray:
        """
        Extract S-parameter data in snp format. The 'n' in 'snp' stands for an
        integer. For instance, s4p stands for scatter parameters
        of a four port device (Scattering 4 Port = s4p).

        For each frequency in the measurement sweep the snp data consists of a
        complex n-by-n matrix. This command returns SNP data without header
        information, and in columns, not in rows as .SnP files. This means that
        the data returned from this command sends all frequency data, then all
        Sx1 magnitude or real data, then all Sx1 phase or imaginary data, and
        so forth.

        For more information about the snp format, please visit:
        http://literature.cdn.keysight.com/litweb/pdf/ads2004a/cktsim/ck04a8.html

        Args:
            ports (list): The ports from which we want data (e.g. [1, 2, 3, 4])

        Returns:
            data (ndarray): Array of length

                (2 * n**2 + 1) * n_freq

            where:
             * n is the number of ports requested
             * n_freq is the number of frequency points in the sweep

        Example:
            Please view:
            qcodes/docs/examples/driver_examples/Qcodes_example_with_Keysight_PNA_N5222B.ipynb
        """
        if ports is None:
            ports_string = "1,2,3,4"
        else:
            ports_string = ",".join([str(p) for p in ports])

        # We want our SNP data in Real-Imaginary format
        self.write('MMEM:STOR:TRAC:FORM:SNP RI')
        write_string = f'CALC{self._channel}:DATA:SNP:PORT? "{ports_string}"'
        data = np.array(
            self.parent.visa_handle.query_binary_values(
                write_string,  datatype='f', is_big_endian=True
            )
        )
        self.parent.synchronize()
        return data

    def __repr__(self):
        return f"<Channel type {self.parent}: {str(self._channel)}>"
