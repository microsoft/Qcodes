import qcodes.instrument_drivers.stanford_research.SR830 as SR830
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils.validators import Numbers, Ints, Enum, MultiType


class Lockin(SR830.SR830):
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, **kwargs)

        for param_name in ['amplitude', 'X', 'Y', 'R']:
            # Replace parameter by parameter_raw
            self.parameters[param_name + '_raw'] = self.parameters[param_name]
            self.parameters.pop(param_name)

        self.add_parameter('amplitude',
                           label='amplitude',
                           get_cmd=lambda: self.amplitude_raw()/self.divider(),
                           set_cmd=lambda ampl: self.amplitude_raw(ampl*self.divider()))

        self.add_parameter('X',
                           get_cmd=lambda: self.X_raw()/self.gain(),
                           set_cmd=lambda x: self.X_raw(x*self.gain()))
        self.add_parameter('Y',
                           get_cmd=lambda: self.Y_raw()/self.gain(),
                           set_cmd=lambda y: self.Y_raw(y*self.gain()))
        self.add_parameter('R',
                           get_cmd=lambda: self.R_raw()/self.gain(),
                           set_cmd=lambda r: self.R_raw(r*self.gain()))

        self.add_parameter('gain',
                           label='Input gain',
                           parameter_class=ManualParameter,
                           initial_value=1
                           )
        self.add_parameter('divider',
                           label='Output divider',
                           parameter_class=ManualParameter,
                           initial_value=1
                           )