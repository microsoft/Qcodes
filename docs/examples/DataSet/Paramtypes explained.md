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

# Paramtypes explained

Internally in the SQLite database on disk, data are registered as being of one of (currently) four allowed types:

- `numeric`
- `array`
- `text`
- `complex`

This notebook seeks to exemplify when each type should be used, and how differently the `Measurement` object treats data of each type.

We start with necessary imports, and then initialising our database and creating an experiment.

```{code-cell} ipython3
import os
import time

import numpy as np

from qcodes import initialise_or_create_database_at, \
    load_or_create_experiment, Measurement, load_by_id
from qcodes.instrument.parameter import ArrayParameter, Parameter
from qcodes.tests.instrument_mocks import DummyInstrument
```

```{code-cell} ipython3
initialise_or_create_database_at(os.path.join(os.getcwd(), 'paramtypes_explained.db'))
exp = load_or_create_experiment('paramtypes', sample_name='not_available')
```

Let us, now, create two dummy instruments to be used in our experiment.

```{code-cell} ipython3
dac = DummyInstrument('dac', gates=['ch1', 'ch2'])
SA = DummyInstrument('SA')
```

```{code-cell} ipython3
# some array-like data types

class Spectrum(ArrayParameter):

    def __init__(self, name, instrument):

        self.N = 7
        setpoints = (np.linspace(0, 1, self.N),)

        super().__init__(name=name,
                         instrument=instrument,
                         setpoints=setpoints,
                         shape=(20,),
                         label='Noisy spectrum',
                         unit='V/sqrt(Hz)',
                         setpoint_names=('Frequency',),
                         setpoint_units=('Hz',))

    def get_raw(self):
        return np.random.randn(self.N)


class MultiDimSpectrum(ArrayParameter):

    def __init__(self, name, instrument):
        self.start = 0
        self.stop = 1
        self.npts = (2, 5, 3)
        sp1 = np.linspace(self.start, self.stop,
                          self.npts[0])
        sp2 = np.linspace(self.start, self.stop,
                          self.npts[1])
        sp3 = np.linspace(self.start, self.stop,
                          self.npts[2])
        setpoints = (sp1,
                     np.tile(sp2, (len(sp1), 1)),
                     np.tile(sp3, (len(sp1), len(sp2), 1)))
        super().__init__(name=name,
                         instrument=instrument,
                         setpoints=setpoints,
                         shape=(100, 50, 20),
                         label='Flower Power Spectrum in 3D',
                         unit='V/sqrt(Hz)',
                         setpoint_names=('Frequency0', 'Frequency1',
                                         'Frequency2'),
                             setpoint_units=('Hz', 'Other Hz', "Third Hz"))
    def get_raw(self):
        a = self.npts[0]
        b = self.npts[1]
        c = self.npts[2]
        return np.reshape(np.arange(a*b*c), (a, b, c))

# a string-valued parameter
def dac1_too_high():
    return 'Too high' if dac.ch1() > 5 else 'OK'
```

Finally, we add our parameters to the dummy instruments:

```{code-cell} ipython3
dac.add_parameter('control', get_cmd=dac1_too_high)
SA.add_parameter('spectrum', parameter_class=Spectrum)
SA.add_parameter('spectrum3D', parameter_class=MultiDimSpectrum)
```

## Numeric

+++

The `numeric` datatype is simply a number. Data registered with this type are saved as individual numbers. This is the **default** datatype when registering parameters.

### Numeric example 1

In this example, all parameters get registered as `numeric` type. This entails that the array in unraveled and inserted point-by-point.

```{code-cell} ipython3
meas = Measurement(exp=exp)
meas.register_parameter(dac.ch1)
meas.register_parameter(SA.spectrum, setpoints=(dac.ch1,))

t0 = time.perf_counter()

with meas.run() as datasaver:
    for dac_v in np.linspace(0, 2, 5):
        dac.ch1(dac_v)
        datasaver.add_result((dac.ch1, dac_v), (SA.spectrum, SA.spectrum()))

t1 = time.perf_counter()

print(f'Finished run in {(t1-t0):.3f} s')

dataset1 = datasaver.dataset
```

The data may be retrieved using the `get_parameter_data` method. This function will bring back the data in a way that reflects the datastructure as it is stored.

```{code-cell} ipython3
dataset1.get_parameter_data()
```

## Array

The `array` paramtype stores data as binary blobs in the database. Insertion is faster (asymptotically **much** faster) this way, but the data are "dead" to SQL queries inside the database. Be informed that a BLOB in sqlite has a default max length limit set at 1 billion (1,000,000,000) bytes (for more information, refer to [Sqlite](https://sqlite.org/limits.html) docs).

+++

### Array example 1

Let us repeat the above measurement, but this time using `array` paramtypes.

```{code-cell} ipython3
meas = Measurement(exp=exp)
meas.register_parameter(dac.ch1)
meas.register_parameter(SA.spectrum, setpoints=(dac.ch1,), paramtype='array')

t0 = time.perf_counter()

with meas.run() as datasaver:
    for dac_v in np.linspace(0, 2, 5):
        dac.ch1(dac_v)
        datasaver.add_result((dac.ch1, dac_v), (SA.spectrum, SA.spectrum()))

t1 = time.perf_counter()

print(f'Finished run in {(t1-t0):.3f} s')

dataset2 = datasaver.dataset
```

```{code-cell} ipython3
dataset2.get_parameter_data()
```

### Array example 2

When storing multidimensional `array` data (think: Alazar cards), both `numeric` and `array` types can be used.

```{code-cell} ipython3
meas = Measurement(exp=exp)
meas.register_parameter(SA.spectrum3D, paramtype='array')

with meas.run() as datasaver:
    datasaver.add_result((SA.spectrum3D, SA.spectrum3D()))
dataset3 = datasaver.dataset
```

The data come out the way we expect them to.

```{code-cell} ipython3
dataset3.get_parameter_data()
```

### Array example 3

For completeness, here, we provide an example where the multidimensional array has an auxiliary setpoint.

```{code-cell} ipython3
meas = Measurement(exp=exp)
meas.register_parameter(dac.ch1)
meas.register_parameter(SA.spectrum3D, paramtype='array', setpoints=(dac.ch1,))

with meas.run() as datasaver:
    for dac_v in [3, 4, 5]:
        dac.ch1(dac_v)
        datasaver.add_result((dac.ch1, dac_v),
                             (SA.spectrum3D, SA.spectrum3D()))
dataset4 = datasaver.dataset
```

```{code-cell} ipython3
dataset4.get_parameter_data()
```

## Text

Text is strings. Sometimes it may be useful to capture categorial data that is represented as string values, or a log message, or else.

Note that the `paramtype` setting is important. The datasaver will not allow to save `numeric` data for a parameter that was registered as `text`. The opposite it also true: the datasaver will not allow to save strings for a parameter what was registered as non-`text` (`numeric` or `array`).

```{code-cell} ipython3
meas = Measurement(exp=exp)
meas.register_parameter(dac.ch1)
meas.register_parameter(dac.control, setpoints=(dac.ch1,), paramtype='text')

with meas.run() as datasaver:
    for dac_v in np.linspace(4, 6, 10):
        dac.ch1(dac_v)
        datasaver.add_result((dac.ch1, dac_v),
                             (dac.control, dac.control()))
dataset5 = datasaver.dataset
```

```{code-cell} ipython3
dataset5.get_parameter_data()
```
