from qcodes import Instrument, Parameter, validators
from nidcpower import Session, OutputFunction, MeasurementTypes, MeasureWhen
import math
from time import time

class _SmuMode(Parameter):
    
    def get_raw(self) -> str:
       if self.parent.connection.output_function.name == "DC_VOLTAGE":
           return "voltage_source"
       elif self.parent.connection.output_function.name == "DC_CURRENT":
           return "current_source"
       else:
           raise ValueError("smu is not in a valid operating mode.")
    
    def set_raw(self, val) -> None:
        if val == "current_source":    
            self.parent.connection.output_function = OutputFunction.DC_CURRENT
            self.parent.connection.measure_when = MeasureWhen.ON_DEMAND
            self.parent.connection.current_level_autorange = True
            self.parent.connection.commit()
        elif val == "voltage_source":
            self.parent.connection.output_function = OutputFunction.DC_VOLTAGE
            self.parent.connection.measure_when = MeasureWhen.ON_DEMAND
            self.parent.connection.voltage_level_autorange = True
            self.parent.connection.commit()
        else:
            raise ValueError('Invalid mode of smu. Valid choices are "current_source" or "voltage_source".'\
                       + ' User input "{}"'.format(val))

class _SmuVoltageLimit(Parameter):
    
    def get_raw(self) -> float:
        return self.parent.connection.voltage_limit

    def set_raw(self, val) -> None:
        self.parent.connection.voltage_limit_range = val
        self.parent.connection.voltage_limit = val
        self.parent.connection.commit()
        
class _SmuCurrentLimit(Parameter):
    
    def get_raw(self) -> float:
        return self.parent.connection.current_limit

    def set_raw(self, val) -> None:
        self.parent.connection.current_limit_range = val
        self.parent.connection.current_limit = val
        self.parent.connection.commit()
        
class _SmuCurrent(Parameter):
    
    def get_raw(self) -> float:
        with self.parent.connection.initiate():
            current = float('nan')
            start = time()
            timeout = self.parent.measurement_timeout
            while math.isnan(current) and (time() - start) < timeout:
                current = self.parent.connection.measure(MeasurementTypes.CURRENT)
        return current

    def set_raw(self, val) -> None:
        if self.parent.mode() != "current_source":
            raise SyntaxError("Must be in current_source mode to set a current")
        with self.parent.connection.initiate():
            self.parent.connection.current_level = val
            self.parent.connection.commit()

class _SmuVoltage(Parameter):
    
    def get_raw(self) -> float:        
        with self.parent.connection.initiate():
            voltage = float('nan')
            start = time()
            timeout = self.parent.measurement_timeout
            while math.isnan(voltage) and (time() - start) < timeout:
                voltage = self.parent.connection.measure(MeasurementTypes.VOLTAGE)
        return voltage
    
    def set_raw(self, val) -> None:
        if self.parent.mode() != "voltage_source":
            raise SyntaxError("Must be in voltage_source mode to set a voltage")
        with self.parent.connection.initiate():
            self.parent.connection.voltage_level = val
            self.parent.connection.commit()

class _SmuCompliance(Parameter):
    
    def get_raw(self) -> bool:
        with self.parent.connection.initiate():
            return not(self.parent.connection.query_in_compliance())

class NISmu_4139(Instrument):
    
    def __init__(self, 
                 name: str,
                 resource_name = "Dev1",
                 options = "DriverSetup=Model:4139; BoardType:PXIe",
                 **kwargs) -> None:
    
        super().__init__(name,**kwargs)
        self.connection = Session(resource_name = resource_name,
                                  channels = 0,
                                  options = options, 
                                  **kwargs)
        self.connection.reset_with_defaults()
        self.connect_message()
        self.measurement_timeout = kwargs.get('measurement_timeout',0.5)
        
        self.add_parameter(name='mode',
                           label='Mode',
                           parameter_class=_SmuMode)
       
        self.add_parameter(name='current_limit',
                           label='Current Limit',
                           unit='A',
                           vals=validators.Numbers(-1,1),
                           parameter_class=_SmuCurrentLimit)
        
        self.add_parameter(name='voltage_limit',
                           label='Voltage Limit',
                           unit='V',
                           vals=validators.Numbers(-24,24),
                           parameter_class=_SmuVoltageLimit)
        
        self.add_parameter(name='current',
                           label='Current',
                           unit='A',
                           parameter_class=_SmuCurrent)

        self.add_parameter(name='voltage',
                           label='Voltage',
                           unit='V',
                           parameter_class=_SmuVoltage)

        self.add_parameter(name='in_compliance',
                           label='In Compliance',
                           parameter_class=_SmuCompliance)
        
        '''Initial settings'''
        self.voltage_limit(10)
        self.current_limit(10E-3)
        self.mode("voltage_source")
        
    def get_idn(self):
        #Returns device metadata for this non-visa device
        return {
                'vendor': self.connection.instrument_manufacturer,
                'model': self.connection.instrument_model,
                'serial': None, # not supported in nidcpower
                'firmware': self.connection.instrument_firmware_revision,
                }
        

    
