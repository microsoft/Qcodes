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

# Legacy parameter examples

These are parameters which are no longer used in the modern implementations of QCoDeS but are kept here as examples for applications which still feature them.

## ArrayParameter
> Note: This is an older base class for array-valued parameters. For any new driver we strongly recommend using `ParameterWithSetpoints` class which is both more flexible and significantly easier to use. Refer to notebook on [writing drivers with ParameterWithSetpoints](Simple-Example-of-ParameterWithSetpoints.ipynb). 

We have kept the documentation shown below of `ArrayParameter` for the legacy purpose.

While storing the `ArrayParameter` data in the database using `datasaver.add_result()` , be informed that it is stored as BLOB in one row of sqlite database. Where the BLOB in sqlite has a default max length limit set at 1 billion (1,000,000,000) bytes. 

`ArrayParameter` is, for now, only gettable.

```{code-cell} ipython3
from qcodes.instrument import ArrayParameter

class ArrayCounter(ArrayParameter):
    def __init__(self):
        # only name and shape are required
        # the setpoints I'm giving here are identical to the defaults
        # this param would get but I'll give them anyway for
        # demonstration purposes
        super().__init__('array_counter', shape=(3, 2),
                         label='Total number of values provided',
                         unit='',
                         # first setpoint array is 1D, second is 2D, etc...
                         setpoints=((0, 1, 2), ((0, 1), (0, 1), (0, 1))),
                         setpoint_names=('index0', 'index1'),
                         setpoint_labels=('Outer param index', 'Inner param index'),
                         docstring='fills a 3x2 array with increasing integers')
        self._val = 0
    
    def get_raw(self):
        # here I'm returning a nested list, but any sequence type will do.
        # tuple, np.array, DataArray...
        out = [[self._val + 2 * i + j for j in range(2)] for i in range(3)]
        self._val += 6
        return out

array_counter = ArrayCounter()

# simple get
print('first call:', array_counter())
```
