import numpy as np
import logging
from typing import cast
import time

from qcodes import InstrumentChannel, Instrument
from qcodes.utils.validators import Numbers, Enum, MultiType

from qcodes.instrument_drivers.Keysight.N52xx.trace import N52xxTrace


logger = logging.getLogger()


class N52xxChannel(InstrumentChannel):
    """
    Allows operations on specific channels.
    """
    def __init__(self, parent: 'Instrument', channel: int):
        super().__init__(parent, f"channel{channel}")

        self._channel = channel

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
            set_cmd=f"SEND{self._channel}:CORR {{}}",
            val_mapping={True: '1', False: '0'}
        )

        self._traces = self._load_traces_from_instrument()

    def _load_traces_from_instrument(self) ->dict:
        """
        Interface to access traces on the instrument

        Returns:
            dict: keys are trace names, values are instance of `N52xxTrace`
        """
        result = self.ask(f"CALC{self._channel}:PAR:CAT:EXT?")
        if result == "NO CATALOG":
            return {}

        trace_info = result.strip("\"").split(",")
        trace_names = trace_info[::2]
        trace_types = trace_info[1::2]

        parent = cast(Instrument, self.parent)

        return {
            name: N52xxTrace(
                parent, self, name, trace_type, present_on_instrument=True
            )
            for name, trace_type in zip(trace_names, trace_types)
        }

    @property
    def trace(self) ->dict:
        """
        List all traces on the instrument
        """
        return {
            name: trace for name, trace in self._traces.items()
            if trace.present_on_instrument
        }

    def add_trace(self, tr_type: str, name: str=None) -> 'N52xxTrace':
        """
        Add a trace the instrument. Note that if a trace with the name given
        already exists on the instrument, this trace is simply returned without
        uploading a second trace with the same name.

        Args:
            name (str): Name of the trace to add
            tr_type (str): Currently only S-parameter types are supported, which
                have the format `Sxy` where x and y are integers.

        Returns:
            trace (N52xxTrace)
        """
        if name is None:
            name = f"CH{self._channel}_{tr_type}"

        trace = self._traces.get(name, None)
        parent = cast(Instrument, self.parent)

        if trace is None:
            trace = N52xxTrace(parent, self, name, tr_type)
            self._traces[name] = trace

        if not trace.present_on_instrument:
            trace.upload_to_instrument()

        trace.select()
        return trace

    def delete_trace(self, name: str) ->None:
        """
        Deletes the trace on the instrument

        Args:
            name (str): Either the name of the trace to delete or "all"
        """
        if name == "all":
            for trace in self.trace.values():
                trace.delete()
        else:
            self.trace[name].delete()

    def select(self) ->None:
        """
        A channel must be selected (active) to modify its settings. A channel
        is selected by selecting a trace on that channel
        """
        traces = list(self.trace.values())
        if len(traces) == 0:
            raise UserWarning("Cannot select channel as no traces have been "
                              "defined ")
        else:
            traces[0].select()

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
        return str(self._channel)
