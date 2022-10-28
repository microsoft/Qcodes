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

# Threaded data acquisition

+++

In this notebook, we will explore how threading can be used with measurement context manager or dond functions for faster data acquisition. It is important to note that, the threading QCoDeS provideds happens per instrument. Meaning, per instrument one thread is created and all parameters from same instrument gets assigned to the same thread for data acquizition. It is generally not safe for more than one thread to communicate with the same instrument at the same time.

Let us begin with some necessary imports.

```{code-cell} ipython3
%matplotlib inline

import numpy as np

import time

from qcodes import (
    load_or_create_experiment,
    initialise_database,
    Measurement,
    Parameter,
)
from qcodes.tests.instrument_mocks import DummyInstrument, DummyInstrumentWithMeasurement
from qcodes.dataset.plotting import plot_dataset
from qcodes.utils.validators import Numbers
from qcodes.utils.threading import ThreadPoolParamsCaller, call_params_threaded
from qcodes.utils.dataset.doNd import do1d
```

Now, setup some instruments!

```{code-cell} ipython3
dac = DummyInstrument('dac', gates=['ch1', 'ch2'])
dmm1 = DummyInstrumentWithMeasurement(name='dmm1', setter_instr=dac)
dmm2 = DummyInstrumentWithMeasurement(name='dmm2', setter_instr=dac)
```

```{code-cell} ipython3
class SleepyDmmExponentialParameter(Parameter):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._ed = self._exponential_decay(5, 0.2)
        next(self._ed)

    def get_raw(self):
        dac = self.root_instrument._setter_instr
        val = self._ed.send(dac.ch1())
        next(self._ed)
        time.sleep(0.1)
        return val

    @staticmethod
    def _exponential_decay(a: float, b: float):
        x = 0
        while True:
            x = yield
            yield a * np.exp(-b * x) + 0.02 * a * np.random.randn()
```

The above parameter class is made to return data with a delay on purpose with help of `time.sleep(0.1)` statement in the `get_raw` method to simulate slow communication with actual instruments. 

```{code-cell} ipython3
dmm1.add_parameter('v3',
                   parameter_class=SleepyDmmExponentialParameter,
                   initial_value=0,
                   label='Gate v3',
                   unit="V",
                   vals=Numbers(-800, 400),
                   get_cmd=None, set_cmd=None)
```

```{code-cell} ipython3
dmm2.add_parameter('v3',
                   parameter_class=SleepyDmmExponentialParameter,
                   initial_value=0,
                   label='Gate v3',
                   unit="V",
                   vals=Numbers(-800, 400),
                   get_cmd=None, set_cmd=None)
```

Initialize the database and load or create an experiment.

```{code-cell} ipython3
initialise_database()
exp = load_or_create_experiment(
    experiment_name='data_acquisition_with_and_without_threads',
    sample_name="no sample"
)
```

## Measurement 1: Non threaded data acquisition

In the following measurment, we do not use threads and note down the time taken for the data acquisition. 

```{code-cell} ipython3
meas1 = Measurement(exp=exp, name='exponential_decay_non_threaded_data_acquisition')
meas1.register_parameter(dac.ch1)
meas1.register_parameter(dmm1.v3, setpoints=(dac.ch1,))
meas1.register_parameter(dmm2.v3, setpoints=(dac.ch1,))
```

```{code-cell} ipython3
data_acq_time = 0
with meas1.run() as datasaver:             
    for set_v in np.linspace(0, 25, 10):
        dac.ch1.set(set_v)
        
        t1 = time.perf_counter()
        datasaver.add_result((dac.ch1, set_v),
                             (dmm1.v3, dmm1.v3.get()),
                             (dmm2.v3, dmm1.v3.get()))
        t2 = time.perf_counter()
        
        data_acq_time += t2 - t1
    
    dataset1D1 = datasaver.dataset
    
print('Report:')
print(f'Data acquisition time:            {data_acq_time} s')
```

```{code-cell} ipython3
ax, cbax = plot_dataset(dataset1D1)
```

## Measurement 2: Threaded data acquisition

In this measurement, we use `ThreadPoolParamsCaller` for threaded data acquisition. Here also we record the time taken for data acquisition.

`ThreadPoolParamsCaller` will create a thread pool, and will call given parameters in those threads. Each group of parameters that have the same ``underlying_instrument`` protperty will be called in it's own separate thread, so that parameters that interact with the same instrument are always called sequentially (since communication within the single instrument is not thread-safe). Thanks to the fact that the pool of threads gets reuse for every new call of the parameters, the performance penalty of creating and shutting down threads is not significant in many cases.

If there is a benefit in creating new threads for every new parameter call, then use ``call_params_threaded`` function instead.

```{code-cell} ipython3
meas2 = Measurement(exp=exp, name='exponential_decay_threaded_data_acquisition')
meas2.register_parameter(dac.ch1)
meas2.register_parameter(dmm1.v3, setpoints=(dac.ch1,))
meas2.register_parameter(dmm2.v3, setpoints=(dac.ch1,))
```

```{code-cell} ipython3
pool_caller = ThreadPoolParamsCaller(dac.ch1, dmm1.v3, dmm2.v3)  # <----- This line is different

data_acq_time = 0
with meas2.run() as datasaver, pool_caller as call_params_in_pool:  # <----- This line is different
    for set_v in np.linspace(0, 25, 10):
        dac.ch1.set(set_v)
        
        t1 = time.perf_counter()
        datasaver.add_result(*call_params_in_pool())  # <----- This line is different
        t2 = time.perf_counter()

        data_acq_time += t2 - t1

        # With ``call_params_threaded`` this line that measures parameters
        # and passes them to the datasaver would be:
        # datasaver.add_result(*call_params_threaded((dac.ch1, dmm1.v3, dmm2.v3)))

    dataset1D2 = datasaver.dataset

print('Report:')
print(f'Data acquisition time:            {data_acq_time} s')
```

```{code-cell} ipython3
ax, cbax = plot_dataset(dataset1D2)
```

## Non threaded and threaded data acquisition with do1d

+++

Lets now see how to do non threaded and threaded data acquisition with `do1d` function. For threaded data acquisition, `use_threads` argument will be set to `True`. Same argument is available on `do0d`, `do2d` and `dond` functions.

### Measurement 3: Non threaded data acquisition with do1d

```{code-cell} ipython3
t0 = time.perf_counter()
do1d(dac.ch1, 0, 1, 10, 0, dmm1.v3, dmm2.v3, do_plot=True)
t1 = time.perf_counter()

print('Report:')
print(f'Data acquisition time:            {t1 - t0} s')
```

###  Measurement 4: Threaded data acquisition with do1d

```{code-cell} ipython3
t0 = time.perf_counter()
do1d(dac.ch1, 0, 1, 10, 0, dmm1.v3, dmm2.v3, do_plot=True, use_threads=True) # <------- This line is different
t1 = time.perf_counter()

print('Report:')
print(f'Data acquisition time:            {t1 - t0} s')
```
