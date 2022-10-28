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

# Parameters in QCoDeS

A `Parameter` is the basis of measurements and control within QCoDeS. Anything that you want to either measure or control within QCoDeS goes thought the `Parameter` interface as it represents the state variables of the system. While many Parameters represent a setting or measurement for a particular `Instrument`, it is possible to define `Parameters` that represent more powerful abstractions.

QCoDeS accomodates setting (i.e. settable parameters) and getting (i.e. gettable parameters) parameter values in a configurable manner. Moreover, various types of data may be accomodated by `Parameters` including simple numbers, strings or a complicated data structure that contains numerical, textual, or other elements.

The value of such a Parameter may be of many types:
- A single numeric value, such as a voltage measurement
- A string that represents a discrete instrument setting, such as the orientation of a vector
- Multiple related values, such as the magnitude and phase or Cartesian components of a vector
- A sequence of values, such as a sampled waveform or a power spectrum
- Multiple sequences of values, such as waveforms sampled on multiple channels
- Any other shape that appropriately represents a characteristic of the Instrument.

## Responsibilities

Parameters have specific responsibilities in QCodes:

- Generating the commands to pass to the Instrument and interpreting its response
- Testing whether an input is valid, via a validator method
- Providing get or set methods for mathematical abstractions
- Providing context and meaning to its data through descriptive attributes (e.g. name, units) 

Parameters hold onto their latest set or measured value via an internal cache, as well as the timestamp of the latest cache update. 

## Examples

In this notebook we provide examples of some parameters and their usage in a QCoDeS environment with dummy instruments. These examples can be used as a starting point to develop parameters and instruments for your applications.


### Imports

```{code-cell} ipython3
import numpy as np
from typing import Optional

from qcodes.instrument.base import (
    InstrumentBase
)
from qcodes.instrument.parameter import (
    Parameter, 
    ArrayParameter,
    MultiParameter, 
    ManualParameter
)
from qcodes.tests.instrument_mocks import (
    DummyInstrument,
)
from qcodes.utils import (
    validators
)
```

### Simple parameter subclass

It is possible to use the `Parameter` class to represent instrument parameters that include custom get and set methods. However, we advise that users be familiar with developing their own parameter sub-classes in order to facilitate instrument communications.

Where the parameter class have user-facing set and get methods, parameter subclasses feature instrument facing set_raw and get_raw methods. This enables instrument inputs and outputs to be parsed and validated from simple method calls, and provides a layer of abstraction between the QCoDeS interface and the physcial device.

+++

#### Parameter definition

```{code-cell} ipython3
class MyCounter(Parameter):
    def __init__(self, name):
        # only name is required
        super().__init__(name, label='Times this has been read',
                         vals=validators.Ints(min_value=0),
                         docstring='counts how many times get has been called '
                                   'but can be reset to any integer >= 0 by set')
        self._count = 0
    
    # you must provide a get method, a set method, or both.
    def get_raw(self):
        self._count += 1
        return self._count
    
    def set_raw(self, val):
        self._count = val
        return self._count
```

#### Demonstration of the get method

```{code-cell} ipython3

c = MyCounter('c')

# c() is equivalent to c.get()
print('Successive calls of c.get():', c(), c(), c(), c(), c())
```

#### Demonstration of the set method

```{code-cell} ipython3
# We can also set the value here
print ('setting counter to 22:', c(22))

print('After, we can get', c())
```

#### Inspecting parameter attributes

When developing protocols, it may be useful to inspect if a parameter is settable or gettable. This can be seen in the respective .settable and .gettable attributes produced by the `Parameter` base class.  

```{code-cell} ipython3
print(f"Is c is gettable? {c.gettable}")
print(f"Is c is settable? {c.settable}")
```

### Virtual Parameters

Users will frequently create a parameter which overlays an existing parameter, creating a further layer of abstraction. This is often done to provide a simple interface to include data processing or validation on top of existing communication infrastructure. We refer to these abstractions as "virtual parameters".


+++

Normally virtual parameters are most easily created using the ``DelegateParameter`` class.

```{code-cell} ipython3
from qcodes.instrument import DelegateParameter
```

First we instantiate our virtual instrument:

```{code-cell} ipython3
dac = DummyInstrument('dac', gates=['ch1', 'ch2'])
dac.ch2.set(1)
dac.print_readable_snapshot()
```

Then we create a DelegateParameter. Note that the DelegateParameter supports changing name, label, unit, scale and offset. Its therefor possible to use a DelegateParameter to perform a simple unit conversion.

```{code-cell} ipython3
my_delegate_param = DelegateParameter('my_delegated_parameter', dac.ch2, scale=1/1000, unit='mV')
```

```{code-cell} ipython3
print(my_delegate_param.get())
print(my_delegate_param.unit)
```

####  Manually creating a virtual parameter

+++

In some cases it may make sense to manually create a virtual parameter. In this example we will create a virtual parameter that abstracts channel 1 of a digital-to-analog converter (`dac`).

+++

We define our virtual parameter to abstract a single parameter of `dac` (i.e. either `ch1` or `ch2`). We will also include a method to return the instance of the underlying instrument (`dac`) from this abstraction.

```{code-cell} ipython3
class VirtualParameter(Parameter):
    def __init__(self, name, dac_param):
        self._dac_param = dac_param
        super().__init__(name)
    
    @property
    def underlying_instrument(self) -> Optional[InstrumentBase]:
        return self._dac_param.root_instrument
    
    def get_raw(self):
        return self._dac_param.get()    
```

> `underlying_insturment`: We advise that this property is included with virtual parameters to avoid race conditions when multi-theading (e.g. using the `dond` function with `use_threads=true`). This allows qcodes to know which instrument in accessed when accessing the parameter. This ensures that a given instrument is ever only accessed from one thread.

Now we will instantiate this to abstract the first channel (`dac.ch1`) using this virtual parameter:

```{code-cell} ipython3
vp1 = VirtualParameter('dac_channel_1', dac.ch1)
```

Notice we have no set method, so we are locked out from accidentally changing the current output voltage.

```{code-cell} ipython3
print(f"Is our virtual parameter is gettable? {vp1.gettable}")
print(f"Is our virtual parameter is settable? {vp1.settable}")
```

#### Instrument Parameters

The most useful `Parameters` are part of an `Instrument`. These `Parameters` are created using the `Instrument's` `add_parameter` method and facilitate low-level communication between QCoDeS and the device.

A settable Parameter typically represents a configuration setting or other controlled characteristic of the Instrument. Most such Parameters have a simple numeric value, but the value can be a string or other data type if necessary. If a settable Parameter is also gettable, getting it typically just reads back the value that was previously set but there can be differences due to processing (e.g. rounding, truncation, etc.).  A Parameter that is only gettable typically represents a single measurement command, and may feature some processing.

These parameters are identical in implementation to the above cases, using set_raw and get_raw methods for instrument facing communications. In order to see examples of these parameters, we advise reviewing our notebooks on insturments and instrument drivers.


```{code-cell} ipython3

```
