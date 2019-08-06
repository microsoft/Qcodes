from numpy import pi
import numpy as np

from qcodes import VisaInstrument, validators as vals
from qcodes.instrument.parameter import MultiParameter

DEBUG = False


class PNA_parameter(MultiParameter):
    def __init__(self, instrument, get_cmd, *args, **kwargs):
        self.ins = instrument
        self.get_cmd = get_cmd
        self.call_func = get_cmd
        self.args = []
        super().__init__(names=self.names, shapes=self.shapes,
                         setpoint_names=(('frequency',),('frequency',)),
                         setpoint_labels=(('Frequency',),('Frequency',)),
                         setpoint_units=(('Hz',),('Hz',)),
                         *args, **kwargs)

    @property
    def shapes(self):
        return ((self.ins.data_dimensions()[1],),) * self.ins.data_dimensions()[0]

    @shapes.setter
    def shapes(self, shapes):
        pass

    @property
    def names(self):
        return self.ins.data_names()

    @names.setter
    def names(self, names):
        pass

    @property
    def setpoints(self):
        return ((tuple(self.ins.frequencies()[:]), ),
                (tuple(self.ins.frequencies()[:]), ))

    @setpoints.setter
    def setpoints(self, vals):
        pass

    def get(self):
        return self.get_cmd()


class Keysight_PNAL(VisaInstrument):
    '''
    This is the qcodes driver for the Keysight PNA L vector network analyzer (VNA)

    This driver does not contain all commands available for the PNA-L but
    only the ones most commonly used.
    '''

    def __init__(self, name, address, timeout=600, **kwargs):
        super().__init__(name, address, timeout=timeout, **kwargs)

        self.add_parameter(name='power',
                           label='Power',
                           unit='dBm',
                           get_cmd='SOURce:POWer?',
                           set_cmd='SOURce:POWer' + ' {:.4f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(-130, 20))

        self.add_parameter(name='IFBW',
                           label='IFBW',
                           unit='Hz',
                           get_cmd='SENSe:BANDwidth?',
                           set_cmd='SENSe:BANDwidth' + ' {:.4f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(1,1e7))

        self.add_parameter(name='frequency_start',
                           label='frequency Start',
                           unit='Hz',
                           get_cmd='SENSe:FREQuency:STARt?',
                           set_cmd='SENSe:FREQuency:STARt ' + ' {:.9f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(300e6, 13.5e9))

        self.add_parameter(name='frequency_stop',
                           label='frequency Stop',
                           unit='Hz',
                           get_cmd='SENSe:FREQuency:STOP?',
                           set_cmd='SENSe:FREQuency:STOP ' + ' {:.9f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(300e6, 13.5e9))

        self.add_parameter(name='frequency_center',
                           label='frequency center',
                           unit='Hz',
                           get_cmd='SENSe:FREQuency:CENTer?',
                           set_cmd='SENSe:FREQuency:CENTer ' + ' {:.9f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(300e6, 13.5e9))

        self.add_parameter(name='frequency_span',
                           label='frequency span',
                           unit='Hz',
                           get_cmd='SENSe:FREQuency:SPAN?',
                           set_cmd='SENSe:FREQuency:SPAN ' + ' {:.9f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(300e6, 13.5e9))

        self.add_parameter(name='frequency_number_of_points',
                           label='Number of points',
                           get_cmd='SENSe:SWEep:POINts?',
                           set_cmd='SENSe:SWEep:POINts ' + ' {:d}',
                           get_parser=int,
                           set_parser=int,
                           vals=vals.Numbers(1, 100001))

        self.add_parameter(name='display_format',
                           label='display format',
                           get_cmd='CALC:FORMat?',
                           set_cmd='CALC:FORMat ' + ' {}',
                           vals=vals.Strings())

        self.add_parameter(name='frequencies',
                           label='frequencies',
                           unit='Hz',
                           get_cmd=lambda: [float(f) for f in self.ask("CALCulate:X?").split(",")],
                           snapshot_value=False)

        self.add_parameter(name='active_trace',
                           label='active trace',
                           set_cmd='CALC:PAR:MNUM ' + ' {:d}')

        self.add_parameter(name='current_trace',
                           parameter_class=PNA_parameter,
                           get_cmd=lambda: self.data_value(new_trace=False))

        self.add_parameter(name='new_trace',
                           parameter_class=PNA_parameter,
                           get_cmd=lambda: self.data_value(new_trace=True))

        self.connect_message()

    def data_value(self, new_trace=False):
        if DEBUG: print("data values...")

        if new_trace:  # if new trace requested, go in single trace and apply a trigger.
            self.write("INITiate:CONTinuous OFF")
            self.ask("INITiate:IMMediate;*OPC?")

        number_of_traces = int(len(self.ask("CALC:PAR:CAT:EXT?")[1:-2].split(',')) / 2)  # number of displayed traces
        data_array = np.zeros(self.data_dimensions())

        for i in range(number_of_traces):
            self.active_trace(i + 1)
            data_array[i, :] = [float(x) for x in self.ask("CALC:DATA? FDATA").split(",")]  # get data and parse

        self.write("INITiate:CONTinuous ON")  # back to continuous tritriggering
        if DEBUG: print("data values OK")
        return data_array

    def get_trace_names(self):
        self._traces = self.ask("CALC:PAR:CAT:EXT?")[1:-2].split(',')[::2]

    def data_dimensions(self):
        if DEBUG: print("data dimensions...")
        r = (int(len(self.ask("CALC:PAR:CAT:EXT?")[1:-2].split(',')) / 2), len(self.frequencies()))
        if DEBUG: print("data dimensions OK")
        return r

    def data_names(self):
        if DEBUG: print("data names...")
        number_of_traces = int(len(self.ask("CALC:PAR:CAT:EXT?")[1:-2].split(',')) / 2)  # number of displayed traces
        names = [''] * (number_of_traces)

        for j in range(number_of_traces):
            self.active_trace(j + 1)
            names[j] = self.display_format()[:-1]
        if DEBUG: print("data names OK")
        return names


