from qcodes.config.config import DotDict
from qcodes.instrument.parameter import _BaseParameter

class ParameterNode():
    parameters = {}

    def __init__(self):
        self.parameters = DotDict()

    def __call__(self) -> dict:
        return self.parameters

    def __getattr__(self, attr):
        if attr in self.parameters:
            return self.parameters[attr]()
        else:
            raise AttributeError(attr)

    def __setattr__(self, attr, val):
        if isinstance(val, _BaseParameter):
            self.parameters[attr] = val
            if val.name == 'None':
                # Parameter created without name, update to attr
                val.name = attr
                if val.label is None:
                    val.label = attr
        elif attr in self.parameters:
            self.parameters[attr](val)
        else:
            super().__setattr__(attr, val)

    def __dir__(self):
        # Add parameters to dir
        items = super().__dir__()
        items.extend(self.parameters.keys())
        return items