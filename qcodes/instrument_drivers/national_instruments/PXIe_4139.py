from qcodes import Instrument, Parameter, validators
from qcodes.instrument.channel import InstrumentChannel

from nidcpower import Session, OutputFunction, MeasurementTypes, MeasureWhen

import math
from time import time

class NiSmu(Instrument):
    def __init__(self,
                 name: str,
                 resource_name = "Dev1",
                 channels = { 0 : 'I' },
                 options = "DriverSetup=Model:4139; BoardType:PXIe",
                 **kwargs) -> None:
        """QCoDeS wrapper for the nidcpower python library from NI
        Basic SMU driver that implements arbitrary channels configured to act
        as either current or voltage sources. NI provide their own connection
        handler that is created on a per instrument basis.
        Args:
            name: the name for the instrument
            resource_name: the name of the resource. this can be found using NIMAX
            channels: a dictonary that maps channel names to source types. e.g.
                default { 0 : 'I' } maps channel 0 to a current source. 'V' is
                used to signify a voltage source.
            options: string that specifies the model number and device type.
                see nidcpower.Session() documentation for more detail
        """

        super().__init__(name, **kwargs)
        channel_str = sum([str(k) + ',' for k in channels.keys()][:-1])
        self.connection = Session(resource_name = resource_name,
                                  channels = channel_str,
                                  options = options, **kwargs)
        self.connection.reset_with_defaults()

        for ch, src_type in channels.items():
            ch_name = f'smu{ch}'
            if src_type == 'I':
                channel = NiSmuCurrentSource(self, ch_name)
            elif src_type == 'V':
                channel = NiSmuVoltageSource(self, ch_name)
            else:
                raise ValueError(f'Source must be I (current) or V (voltage). \
                                   Received {ch}')
            self.add_submodule(ch_name, channel)

        self.connect_message()

        if 'measurement_timeout' in kwargs:
            self.measurement_timeout = kwargs['measurement_timeout']
        else:
            self.measurement_timeout = 0.500

    def get_idn(self):
        """Returns device metadata for this non-visa device"""
        return {
            'vendor': self.connection.instrument_manufacturer,
            'model': self.connection.instrument_model,
            'serial': None, # not supported in nidcpower
            'firmware': self.connection.instrument_firmware_revision,
        }

class NiSmuCurrentSource(InstrumentChannel):
    def __init__(self, parent: Instrument, name: str) -> None:
        """Utilises an SMU channel as a current source"""
        super().__init__(parent, name)

        self.parent.connection.output_function = OutputFunction.DC_CURRENT
        self.parent.connection.measure_when = MeasureWhen.ON_DEMAND
        self.parent.connection.current_level_autorange = True
        self.parent.connection.commit()

        self.add_parameter(name='current',
                           initial_value=0,
                           label='Current',
                           unit='A',
                           vals=validators.Numbers(-3,3),
                           parameter_class=NiSmuCurrentSource._SmuCurrent)

        self.add_parameter(name='voltage',
                           label='Voltage',
                           unit='V',
                           parameter_class=NiSmuCurrentSource._SmuVoltage)

        self.add_parameter(name='voltage_limit',
                           initial_value=24,
                           label='Voltage Limit',
                           unit='V',
                           vals=validators.Numbers(-60,60),
                           parameter_class=NiSmuCurrentSource._SmuVoltageLimit)

        self.add_parameter(name='current_limit',
                           label='Current Limit',
                           unit='A',
                           parameter_class=NiSmuCurrentSource._SmuCurrentLimit)

        self.add_parameter(name='in_compliance',
                           label='In Compliance',
                           parameter_class=_SmuCompliance)

    class _SmuCurrent(Parameter):
        def get_raw(self) -> float:
            with self.instrument.parent.connection.initiate():
                current = float('nan')
                start = time()
                timeout = self.instrument.parent.measurement_timeout
                while math.isnan(current) and (time() - start) < timeout:
                    current = self.instrument.parent.connection.measure(MeasurementTypes.CURRENT)
            return current

        def set_raw(self, val) -> None:
            with self.instrument.parent.connection.initiate():
                self.instrument.parent.connection.current_level = val
                self.instrument.parent.connection.commit()

    class _SmuVoltage(Parameter):
        def get_raw(self) -> float:
            with self.instrument.parent.connection.initiate():
                voltage = float('nan')
                start = time()
                timeout = self.instrument.parent.measurement_timeout
                while math.isnan(voltage) and (time() - start) < timeout:
                    voltage = self.instrument.parent.connection.measure(MeasurementTypes.VOLTAGE)
            return voltage

    class _SmuCurrentLimit(Parameter):
        def get_raw(self) -> float:
            return self.instrument.parent.connection.current_level_range

    class _SmuVoltageLimit(Parameter):
        def get_raw(self) -> float:
            return self.connection.instrument.voltage_limit

        def set_raw(self, val) -> None:
            self.instrument.parent.connection.voltage_limit_range = val
            self.instrument.parent.connection.voltage_limit = val
            self.instrument.parent.connection.commit()

class NiSmuVoltageSource(InstrumentChannel):
    def __init__(self, parent: Instrument, name: str) -> None:
        """Utilises an SMU channel as a voltage source"""
        super().__init__(parent, name)

        self.parent.connection.output_function = OutputFunction.DC_VOLTAGE
        self.parent.connection.measure_when = MeasureWhen.ON_DEMAND
        self.parent.connection.voltage_level_autorange = True
        self.parent.connection.commit()

        self.add_parameter(name='voltage',
                           initial_value=0,
                           label='Voltage',
                           unit='V',
                           vals=validators.Numbers(-50,50),
                           parameter_class=NiSmuVoltageSource._SmuVoltage)

        self.add_parameter(name='current',
                           label='Current',
                           unit='A',
                           parameter_class=NiSmuVoltageSource._SmuCurrent)

        self.add_parameter(name='voltage_limit',
                           label='Voltage Limit',
                           unit='V',
                           parameter_class=NiSmuVoltageSource._SmuVoltageLimit)

        self.add_parameter(name='current_limit',
                           initial_value=10e-3,
                           label='Current Limit',
                           unit='A',
                           vals=validators.Numbers(-3,3),
                           parameter_class=NiSmuVoltageSource._SmuCurrentLimit)

        self.add_parameter(name='in_compliance',
                           label='In Compliance',
                           parameter_class=_SmuCompliance)

    class _SmuCurrent(Parameter):
        def get_raw(self) -> float:
            with self.instrument.parent.connection.initiate():
                current = float('nan')
                start = time()
                timeout = self.instrument.parent.measurement_timeout
                while math.isnan(current) and (time() - start) < timeout:
                    current = self.instrument.parent.connection.measure(MeasurementTypes.CURRENT)
            return current

    class _SmuVoltage(Parameter):
        def get_raw(self) -> float:
            with self.instrument.parent.connection.initiate():
                voltage = float('nan')
                start = time()
                timeout = self.instrument.parent.measurement_timeout
                while math.isnan(voltage) and (time() - start) < timeout:
                    voltage = self.instrument.parent.connection.measure(MeasurementTypes.VOLTAGE)
            return voltage

        def set_raw(self, val) -> None:
            with self.instrument.parent.connection.initiate():
                self.instrument.parent.connection.voltage_level = val
                self.instrument.parent.connection.commit()

    class _SmuCurrentLimit(Parameter):
        def get_raw(self) -> float:
            return self.instrument.parent.connection.current_limit

        def set_raw(self, val) -> None:
            self.instrument.parent.connection.current_limit_range = val
            self.instrument.parent.connection.current_limit = val
            self.instrument.parent.connection.commit()

    class _SmuVoltageLimit(Parameter):
        def get_raw(self) -> float:
            return self.instrument.parent.connection.voltage_level_range

class _SmuCompliance(Parameter):
    def get_raw(self) -> bool:
        with self.instrument.parent.connection.initiate():
            return not(self.instrument.parent.connection.query_in_compliance())