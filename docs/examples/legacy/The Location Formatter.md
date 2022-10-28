---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# The Location Formatter

The Location Formatter controls the format of the location to which data are saved.

This notebook shows some examples of setting different location formats.

```{code-cell} ipython3
%matplotlib nbagg
import matplotlib.pyplot as plt
import time
import numpy as np

import qcodes as qc
from qcodes.loops import Loop
from qcodes.data.location import FormatLocation
```

```{code-cell} ipython3
# First we set up some mock experiment
from qcodes.tests.instrument_mocks import DummyInstrument

gates = DummyInstrument('some_gates', gates=['plunger', 'left', 'topo'])
meter = DummyInstrument('meter', gates=['voltage', 'current'])

station = qc.Station(gates, meter)
```

## The formatter in action

Now let's run some loops to get datasets and see where they end up.

When writing the location format, some fields are automatically filled out.

That is the fields '{date}', '{time}', and '{counter}'.
All other fields must have their values provided via the record dict.

```{code-cell} ipython3
loc_fmt='{date}/#{counter}_{name}_{date}_{time}'  # set the desired location format
rcd={'name': 'unicorn'}  # provide a value for 'name'
loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)  # create a location provider using that format

loop = Loop(gates.plunger.sweep(0, 1, num=25), 0).each(meter.voltage)
data2 = loop.run(location=loc_provider)
```

```{code-cell} ipython3
# Now let's do that a few times with different formats

import numpy as np

loc_fmt='my_custom_folder/#{counter}_randomnumber_{name}_{date}_{time}'
rcd = {'name': str(np.random.randint(1, 100))}
loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)

loop = Loop(gates.plunger.sweep(0, 1, num=25), 0).each(meter.voltage)
data2 = loop.run(location=loc_provider)
```

```{code-cell} ipython3
# You can also overwrite the custom fields

loc_fmt='{date}/#{counter}_{name}_{date}_{time}'
rcd = {'time': 'hammer_time'}
loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)

loop = Loop(gates.plunger.sweep(0, 1, num=25), 0).each(meter.voltage)
data2 = loop.run(location=loc_provider)
```

```{code-cell} ipython3

```
