"""Mock instruments for testing purposes."""

from .parameter import MultiParameter
from qcodes import Loop
from qcodes.data.data_array import DataArray


class ArrayGetter(MultiParameter):
    """
    Example parameter that just returns a single array

    TODO: in theory you can make this an ArrayParameter with
    name, label & shape (instead of names, labels & shapes) and altered
    setpoints (not wrapped in an extra tuple) and this mostly works,
    but when run in a loop it doesn't propagate setpoints to the
    DataSet. This is a bug
    """
    def __init__(self, measured_param, sweep_values, delay):
        name = measured_param.name
        super().__init__(names=(name,),
                         shapes=((len(sweep_values),),),
                         name=name)
        self._instrument = getattr(measured_param, '_instrument', None)
        self.measured_param = measured_param
        self.sweep_values = sweep_values
        self.delay = delay
        self.shapes = ((len(sweep_values),),)
        set_array = DataArray(parameter=sweep_values.parameter,
                              preset_data=sweep_values)
        self.setpoints = ((set_array,),)
        if hasattr(measured_param, 'label'):
            self.labels = (measured_param.label,)

    def get(self):
        loop = Loop(self.sweep_values, self.delay).each(self.measured_param)
        data = loop.run_temp()
        array = data.arrays[self.measured_param.full_name]
        return (array,)
