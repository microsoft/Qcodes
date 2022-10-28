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

# Measure without a Loop

If you have a parameter that returns a whole array at once, often you want to measure it directly into a DataSet.

This shows how that works in QCoDeS

```{code-cell} ipython3
%matplotlib nbagg
import qcodes as qc
import numpy as np
# import dummy driver for the tutorial
from qcodes.tests.instrument_mocks import DummyInstrument, DummyChannelInstrument
from qcodes.measure import Measure
from qcodes.actions import Task

dac1 = DummyInstrument(name="dac")
dac2 = DummyChannelInstrument(name="dac2")


# the default dummy instrument returns always a constant value, in the following line we make it random
# just for the looks 💅
dac2.A.dummy_array_parameter.get =  lambda: np.random.randint(0, 100, size=5)

# The station is a container for all instruments that makes it easy
# to log meta-data
station = qc.Station(dac1, dac2)
```

## Instantiates all the instruments needed for the demo

For this tutorial we're going to use the regular parameters (c0, c1, c2, vsd) and ArrayGetter, which is just a way to construct a parameter that returns a whole array at once out of simple parameters, as well as AverageAndRaw, which returns a scalar *and* an array together.

+++

### Only array output
The arguments to Measure are all the same actions you use in a Loop.
If they return only arrays, you will see exactly those arrays (with their setpoints) in the output DataSet

```{code-cell} ipython3
data = Measure(
    Task(dac1.dac1.set, 0),
    dac2.A.dummy_array_parameter,
    Task(dac1.dac1.set, 2),
    dac2.A.dummy_array_parameter,
).run()
```
