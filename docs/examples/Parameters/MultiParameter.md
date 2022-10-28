---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

## MultiParameter
Return multiple items at once, where each item can be a single value or an array. 

> Note: Most of the kwarg names here are the plural of those used in `Parameter` and `ArrayParameter`. In particular, `MultiParameter` is the ONLY one that uses `units`, all the others use `unit`.

`MultiParameter` is, for now, only gettable.

```{code-cell} ipython3
import numpy as np

from qcodes.instrument.parameter import MultiParameter, ManualParameter
```

```{code-cell} ipython3
class SingleIQPair(MultiParameter):
    def __init__(self, scale_param):
        # only name, names, and shapes are required
        # this version returns two scalars (shape = `()`)
        super().__init__('single_iq', names=('I', 'Q'), shapes=((), ()),
                         labels=('In phase amplitude', 'Quadrature amplitude'),
                         units=('V', 'V'),
                         # including these setpoints is unnecessary here, but
                         # if you have a parameter that returns a scalar alongside
                         # an array you can represent the scalar as an empty sequence.
                         setpoints=((), ()),
                         docstring='param that returns two single values, I and Q')
        self._scale_param = scale_param
    
    def get_raw(self):
        scale_val = self._scale_param()
        return (scale_val, scale_val / 2)

scale = ManualParameter('scale', initial_value=2)
iq = SingleIQPair(scale_param=scale)

# simple get
print('simple get:', iq())
```

```{code-cell} ipython3
class IQArray(MultiParameter):
    def __init__(self, scale_param):
        # names, labels, and units are the same 
        super().__init__('iq_array', names=('I', 'Q'), shapes=((5,), (5,)),
                         labels=('In phase amplitude', 'Quadrature amplitude'),
                         units=('V', 'V'),
                         # note that EACH item needs a sequence of setpoint arrays
                         # so a 1D item has its setpoints wrapped in a length-1 tuple
                         setpoints=(((0, 1, 2, 3, 4),), ((0, 1, 2, 3, 4),)),
                         docstring='param that returns two single values, I and Q')
        self._scale_param = scale_param
        self._indices = np.array([0, 1, 2, 3, 4])

    def get_raw(self):
        scale_val = self._scale_param()
        return (self._indices * scale_val, self._indices * scale_val / 2)

iq_array = IQArray(scale_param=scale)

# simple get
print('simple get', iq_array())
```
