from qcodes.instrument_drivers.Keysight.M3300A import Keysight_M3300A_FPGA
from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.utils.validators import Bool, Ints

import numpy as np

try:
    import keysightSD1
except ImportError:
    raise ImportError('to use the Keysight SD drivers install the keysightSD1 module '
                      '(http://www.keysight.com/main/software.jspx?ckey=2784055)')

model_channels = {'M3201A': 4,
                  'M3300A': 4}


class PCDDSChannel(InstrumentChannel):
    """
    Class that holds all the actions related to a specific PCDDS channel
    """
    def __init__(self, parent: Instrument, name: str, id: int, **kwargs):
        super().__init__(parent=parent, name=name, **kwargs)
        self.fpga = self._parent.fpga
        self.id = id
        self.n_pointer_bits = 9
        self.n_op_bits = 10
        self.n_phase_bits = 45
        self.n_accum_bits = 45
        self.n_amp_bits = 16
        self.clk = 100e6
        self.v_max = 3.0
        self.f_max = 200e6

        self.add_parameter(
            'output_enable',
            label=f'ch{self.id} output_enable',
            set_cmd=self._set_output_enable,
            vals=Bool(),
            docstring='Whether the system has an enabled output'
        )

        self.add_parameter(
            'load_delay',
            label=f'ch{self.id} load_delay',
            set_cmd=self._set_load_delay,
            vals=Ints(0, 15),
            docstring='How long the delay should be during loading a new pulse to calculate the new coefficients'
        )

        self.add_parameter(
            'pcdds_enable',
            label=f'ch{self.id} pcdds_enable',
            set_cmd=self._set_pcdds_enable,
            vals=Bool(),
            docstring='Set the output of the device to use the PCDDS system and not the inbuilt functionality'
        )

    def _set_output_enable(self, output_enable):
        """
        Set the output enable state of the module
        Args:
            output_enable: (Bool) Is the output enabled
        """
        operation = int('0010000000', 2)
        if output_enable:
            operation += int('0001000000', 2)
        instr = self.construct_instruction(operation, 0)
        self.fpga.set_fpga_pc_port(self.id, [instr], 0, 0, 1)

    def _set_pcdds_enable(self, pcdds_enable):
        """
        Select the output between the PCDDS system and the passthrough connection
        Args:
            pcdds_enable: (Bool) If true, select the PCDDS system. Otherwise use passthrough
        """
        operation = int('0100000000', 2)
        if pcdds_enable:
            operation += int('0000000001', 2)
        instr = self.construct_instruction(operation, 0)
        self.fpga.set_fpga_pc_port(self.id, [instr], 0, 0, 1)

    def _set_load_delay(self, delay):
        """
        Set the delay that the system will apply to calculate all the phase and frequency coefficiences
        Args:
            delay: (Int) Delay in clock cycles
        """
        operation = int('0000010000', 2) + delay
        # Construct and send instruction
        instr = self.construct_instruction(operation, 0)
        self.fpga.set_fpga_pc_port(self.id, [instr], 0, 0, 1)

    def construct_instruction(self, operation, pointer):
        """
        Function to construct the int instruction packet from the operation and pointer
        Args:
            operation: (Int) Operation that we want to do
            pointer: (Int) ID of the pulse that this instruction refers to

        Returns: (Int) Instruction

        """
        # Check that the pointer is in the allowed range
        if pointer >= 2**self.n_pointer_bits:
            raise ValueError(
                'Pointer with value {} is outside of bounds [{}, {}]'.format(pointer, 0, 2**self.n_pointer_bits-1))
        # Convert and return
        return (operation << 22) + pointer

    def write_zero_pulse(self, pulse):
        """
        Function to write all zeros to a specific memory location
        Args:
            pulse: (Int) The location where the zeros are to be written
        """
        self.write_pulse(pulse=pulse, phase=0, frequency=0, frequency_accumulation=0, amplitude=0, next_pulse=0)

    def clear_memory(self):
        """
        Function to clear all memory
        """
        for i in np.arange(2**self.n_pointer_bits):
            self.write_zero_pulse(pulse=i)

    def write_instr(self, instr):
        """
        Function to write an instruction to the pulse memory.

        It takes a dictionary with the following entries:
        - instr: The instruction type. Valid instruction types are 'dc', 'sine' and 'chrip'
        - pulse_idx: The pulse id that this instruction is to be stored in
        - phase: The phase of the relevant pulse. Only applicable to 'sine' and 'chirp' pulses
        - freq: The frequency of this pulse. Only applicable to 'sine' and 'chirp' pulses
        - accum: The frequency accumulation rate. Only applicable to 'chirp' pulses
        - amp: The amplitude of the pulse
        - next_pulse: The next pulse that should be played after this one
        Args:
            instr: (Dict) The instruction to write to memory
        """
        if instr['instr'] == 'dc':
            self.write_dc_pulse(pulse=instr['pulse_idx'], voltage=instr['amp'], next_pulse=instr['next_pulse'])
        elif instr['instr'] == 'sine':
            self.write_sine_pulse(pulse=instr['pulse_idx'],
                                  phase=instr['phase'],
                                  frequency=instr['freq'],
                                  amplitude=instr['amp'],
                                  next_pulse=instr['next_pulse'])
        elif instr['instr'] == 'chirp':
            self.write_chirp_pulse(pulse=instr['pulse_idx'],
                                   phase=instr['phase'],
                                   frequency=instr['freq'],
                                   frequency_accumulation=instr['accum'],
                                   amplitude=instr['amp'],
                                   next_pulse=instr['next_pulse'])
        else:
            raise ValueError('Unknown instruction type: {}'.format(instr['instr']))

    def write_sine_pulse(self, pulse, phase, frequency, amplitude, next_pulse):
        """
        Write a normal sinusoidal pulse with the desired properties to the pulse memory.
        Args:
            pulse: (Int) The location in pulse memory that this is to be written to
            phase: (Float) The phase of the signal in degrees
            frequency: (Float) The frequency of the signal in Hz
            amplitude: (Float) The desired output maximum amplitude in V
            next_pulse: (Int) The next pulse that the system is to go to after this one
        """
        if not isinstance(pulse, int):
            raise TypeError('Incorrect type for function input pulse. It should be an int')
        if not isinstance(next_pulse, int):
            raise TypeError('Incorrect type for function input next_pulse. It should be an int')
        # Convert all the pulse parameters to the correct register values
        phase_val = self.phase2val(phase)
        freq_val = self.freq2val(frequency)
        accum_val = 0
        amplitude_val = self.amp2val(amplitude)
        self.write_pulse(pulse, phase_val, freq_val, accum_val, amplitude_val, next_pulse)

    def write_dc_pulse(self, pulse, voltage, next_pulse):
        """
        Write a DC pulse to memory. This sets up a pulse with a phase offset of 90 or 270 degrees, 0 frequency
        and a certain amplitude
        Args:
            pulse: (Int) The location in pulse memory that this is to be written to
            voltage: (Float) The desired DC voltage
            next_pulse: (Int) The next pulse that the system is to go to after this one
        """
        if not isinstance(pulse, int):
            raise TypeError('Incorrect type for function input pulse. It should be an int')
        if not isinstance(next_pulse, int):
            raise TypeError('Incorrect type for function input next_pulse. It should be an int')
        # Convert the voltage to the correct register value
        amplitude_val = self.amp2val(np.abs(2.0*voltage))
        if voltage < 0:
            phase_val = self.phase2val(270.0)
        else:
            phase_val = self.phase2val(90.0)
        accum_val = 0
        freq_val = 0
        self.write_pulse(pulse, phase_val, freq_val, accum_val, amplitude_val, next_pulse)

    def write_chirp_pulse(self, pulse, phase, frequency, frequency_accumulation, amplitude, next_pulse):
        """
        Write a pulse to pulse memory which contains a frequency sweep
        Args:
            pulse: (Int) The location in pulse memory that this is to be written to
            phase: (Float) The phase value for this pulse in degrees
            frequency: (Float) The frequency value for this pulse in Hz
            frequency_accumulation: (Float) The frequency accumulation for this pulse in Hz/s
            amplitude: (Float) The amplitude for this pulse in V
            next_pulse: (Int) The pulse that the system is to go to after this one
        """
        if not isinstance(pulse, int):
            raise TypeError('Incorrect type for function input pulse. It should be an int')
        if not isinstance(next_pulse, int):
            raise TypeError('Incorrect type for function input next_pulse. It should be an int')
        phase_val = self.phase2val(phase)
        freq_val = self.freq2val(frequency)
        accum_val = self.accum2val(frequency_accumulation)
        amplitude_val = self.amp2val(amplitude)
        self.write_pulse(pulse, phase_val, freq_val, accum_val, amplitude_val, next_pulse)

    def write_pulse(self, pulse, phase, frequency, frequency_accumulation, amplitude, next_pulse):
        """
        Function to write a pulse with given register values to a given location in pulse memory
        Args:
            pulse: (Int) The location in pulse memory that this is to be written to
            phase: (Int) The phase register value for this pulse
            frequency: (Int) The frequency register value for this pulse
            frequency_accumulation: (Int) The frequency accumulation register for this pulse
            amplitude: (Int) The amplitude register for this pulse
            next_pulse: (Int) The pulse that the system is to go to after this one
        """
        if pulse < 0 or pulse > 2**self.n_pointer_bits:
            raise ValueError(
                'The pulse index is outside of memory. It should be between 0 and {}'.format(2**self.n_pointer_bits-1))
        if next_pulse < 0 or next_pulse > 2**self.n_pointer_bits:
            raise ValueError(
                'The next_pulse index is outside of memory. It should be between 0 and {}'.format(
                    2 ** self.n_pointer_bits - 1))
        # Construct the initial instruction to write a new pulse to memory
        operation = int('0000100000', 2)
        instr = self.construct_instruction(operation, pulse)
        # Construct the pulse parameter to be written to memory
        pulse_data = phase
        pulse_data += (frequency << self.n_phase_bits)
        pulse_data += (frequency_accumulation << 2 * self.n_phase_bits)
        pulse_data += (amplitude << 2 * self.n_phase_bits + self.n_accum_bits)
        pulse_data += (next_pulse << 2 * self.n_phase_bits + self.n_accum_bits + self.n_amp_bits)
        pulse_data = self.split_value(pulse_data)
        self.fpga.set_fpga_pc_port(self.id, [instr], 0, 0, 1)
        self.fpga.set_fpga_pc_port(self.id, [pulse_data[4]], 0, 0, 1)
        self.fpga.set_fpga_pc_port(self.id, [pulse_data[3]], 0, 0, 1)
        self.fpga.set_fpga_pc_port(self.id, [pulse_data[2]], 0, 0, 1)
        self.fpga.set_fpga_pc_port(self.id, [pulse_data[1]], 0, 0, 1)
        self.fpga.set_fpga_pc_port(self.id, [pulse_data[0]], 0, 0, 1)

    @staticmethod
    def split_value(value):
        """
        Splits a 20 byte message up into 5x 32 bit messages
        Args:
            value: (Int) The message that is to be split

        Returns: (List of Ints) List of 32 bit length ints to be sent as messages
        """
        if not isinstance(value, int):
            raise TypeError('Incorrect type passed to split_value')
        return [int(value & 0xFFFFFFFF),
                int((value >> 32) & 0xFFFFFFFF),
                int((value >> 64) & 0xFFFFFFFF),
                int((value >> 96) & 0xFFFFFFFF),
                int((value >> 128) & 0xFFFFFFFF)]

    def set_next_pulse(self, pulse, update):
        """
        Function to set the next pulse to be played. It is also possible to update to this new pulse
        via this function
        Args:
            pulse: (Int) Next pulse to be played
            update: (Bool) Should the system update right now
        """
        operation = int('0000000010', 2)
        if update:
            operation += int('1000000000', 2)
        instr = self.construct_instruction(operation, pulse)
        self.fpga.set_fpga_pc_port(self.id, [instr], 0, 0, 1)

    def send_trigger(self):
        """
        Send a trigger signal to the FPGA
        """
        operation = int('1000000000', 2)
        instr = self.construct_instruction(operation, 0)
        self.fpga.set_fpga_pc_port(self.id, [instr], 0, 0, 1)

    def phase2val(self, phase):
        """
        Function to calculate the correct phase register values for a given phase
        Args:
            phase: (Float) The desired phase in degrees.

        Returns: (Int) The register value for the desired phase
        """
        phase = phase % 360.0
        return int(np.round((2 ** self.n_phase_bits / 360.0) * phase))

    def freq2val(self, freq):
        """
        Function to calculate the correct frequency register values for a given frequency
        Args:
            freq: (Float) The desired frequency in Hz

        Returns: (Int) The register value for the desired frequency
        """
        if freq > self.f_max or freq < 0:
            raise ValueError('Frequency of {0} is outside of allowed values [0, {1}MHz]'.format(freq, self.f_max/1e6))
        return int(np.round((2 ** self.n_phase_bits / (5*self.clk)) * freq))

    def accum2val(self, accum):
        """
        Function to calculate the correct accumulation register values for a given accumulation
        Args:
            accum: (Float) The desired accumulation in Hz/s

        Returns: (Int) The register value for the desired accumulation
        """
        if accum < 0 or accum > (5*self.clk ** 2):
            raise ValueError('Frequency Accumulation of {0} is outside of allowed values [0,{1}Hz/s]'.format(
                accum, (5*self.clk ** 2)))
        return int(np.round(accum * 2 ** self.n_accum_bits / (5*self.clk) ** 2))

    def amp2val(self, amp):
        """
        Function to calculate the correct amplitude register values for a given amplitude
        Args:
            amp: (Float) The desired amplitude in V

        Returns: (Int) The register value for the desired amplitude
        """
        if amp < 0 or amp > self.v_max:
            raise ValueError('Amplitude of {0} is outside of allowed values [0, {1}V'.format(amp, self.v_max))
        return int(np.round((2**self.n_amp_bits-1) * amp/self.v_max))


class PCDDS(Instrument):
    """
    This class is the driver for the Phase Coherent Pulse Generation Module implemented on the FPGA onboard a Keysight
    PXI AWG card
    """
    def __init__(self, name, model, chassis, slot, channels=None, triggers=8, **kwargs):
        """ Constructor for the pulse generation modules """
        super().__init__(name, model, chassis, slot, triggers, **kwargs)
        # TODO: Convert to SD_FPGA, have a look at FPGA_TriggerController
        self.fpga = Keysight_M3300A_FPGA('FPGA')

        if channels is None:
            channels = model_channels[self.model]
        self.n_channels = channels

        channels = ChannelList(self,
                               name='channels',
                               chan_type=PCDDSChannel)
        for ch in range(self.n_channels):
            channel = PCDDSChannel(self, name=f'ch{ch}', id=ch)
            setattr(self, f'ch{ch}', channel)
            channels.append(channel)
        self.add_submodule('channels', channels)

    def reset(self):
        """ Sends the reset signal to the FPGA """
        self.fpga.reset(reset_mode=keysightSD1.SD_ResetMode.PULSE)