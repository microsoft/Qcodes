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

# A ParameterWithSetpoints Example with Dual Setpoints

This notebook explains how you can account for dual setpoints using `ParameterWithSetpoints`. The basics of writing drivers using `ParameterWithSetpoints` is covered in the notebook named [Simple Example of ParameterWithSetpoints](../Parameters/Simple-Example-of-ParameterWithSetpoints.ipynb).

In this example we consider a dummy instrument that can return a time trace or the discreet Fourier transform (magnitude square) of that trace. The setpoints are accounted for in an easy way.

```{code-cell} ipython3
import os
import numpy as np
import matplotlib.pyplot as plt


from qcodes import Measurement, initialise_or_create_database_at, load_or_create_experiment, load_by_id
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ParameterWithSetpoints, Parameter
from qcodes.dataset.plotting import plot_dataset
import qcodes.utils.validators as vals
```

```{code-cell} ipython3
def timetrace(npts: int, dt: float) -> np.ndarray:
    """
    A very realistic-looking signal
    """
    #freq = 10/(dt*npts)
    #decay = 1/(dt*npts)
    freq = 10
    decay = 1
    time = np.linspace(0, npts*dt, npts, endpoint=False)
    signal = np.exp(-decay*time)*np.sin(2*np.pi*freq*time)
    noise = 0.1*np.random.randn(npts)
    return signal + noise
```

```{code-cell} ipython3


class TimeTrace(ParameterWithSetpoints):
    
    def get_raw(self):
        npts = self.root_instrument.npts()
        dt = self.root_instrument.dt()
        
        return timetrace(npts, dt)
    

class Periodogram(ParameterWithSetpoints):
    
    def get_raw(self):
        npts = self.root_instrument.npts()
        dt = self.root_instrument.dt()
        
        tt = self.root_instrument.trace()
        
        return np.abs(np.fft.fft(tt))**2
        
        
class TimeAxis(Parameter):

    def get_raw(self):
        npts = self.root_instrument.npts()
        dt = self.root_instrument.dt()
        return np.linspace(0, dt*npts, npts, endpoint=False)

    
class FrequencyAxis(Parameter):
    
    def get_raw(self):
        npts = self.root_instrument.npts()
        dt = self.root_instrument.dt()

        return np.linspace(0, 1/dt, npts)
    
        
class OzzyLowScope(Instrument):
    
    def __init__(self, name, **kwargs):
        
        super().__init__(name, **kwargs)
        
        self.add_parameter(name='npts',
                           initial_value=500,
                           label='Number of points',
                           get_cmd=None,
                           set_cmd=None)
        
        self.add_parameter(name='dt',
                           initial_value=1e-3,
                           label='Time resolution',
                           unit='s',
                           get_cmd=None,
                           set_cmd=None)
        
        self.add_parameter(name='time_axis',
                           label='Time',
                           unit='s',
                           vals=vals.Arrays(shape=(self.npts,)),
                           parameter_class=TimeAxis)
        
        self.add_parameter(name='freq_axis',
                           label='Frequency',
                           unit='Hz',
                           vals=vals.Arrays(shape=(self.npts,)),
                           parameter_class=FrequencyAxis)
        
        self.add_parameter(name='trace',
                           label='Signal',
                           unit='V',
                           vals=vals.Arrays(shape=(self.npts,)),
                           setpoints=(self.time_axis,),
                           parameter_class=TimeTrace)
        
        self.add_parameter(name='periodogram',
                           label='Periodogram',
                           unit='V^2/Hz',
                           vals=vals.Arrays(shape=(self.npts,)),
                           setpoints=(self.freq_axis,),
                           parameter_class=Periodogram)
```

```{code-cell} ipython3
osc = OzzyLowScope('osc')
```

```{code-cell} ipython3
tutorial_db_path = os.path.join(os.getcwd(), 'tutorial_doND.db')
initialise_or_create_database_at(tutorial_db_path)
load_or_create_experiment(experiment_name='tutorial_exp', sample_name="no sample")
```

## Measurement 1: Time Trace

```{code-cell} ipython3
timemeas = Measurement()
timemeas.register_parameter(osc.trace)

osc.dt(0.001)

with timemeas.run() as datasaver:
    datasaver.add_result((osc.trace, osc.trace.get()))
    
dataset = datasaver.dataset
```

```{code-cell} ipython3
_ = plot_dataset(dataset)
```

```{code-cell} ipython3
osc.dt(0.01)  # make the trace 10 times longer

with timemeas.run() as datasaver:
    datasaver.add_result((osc.trace, osc.trace.get()))
    
dataset = datasaver.dataset
```

```{code-cell} ipython3
_ = plot_dataset(dataset)
```

## Measurement 2: Periodogram

```{code-cell} ipython3
freqmeas = Measurement()
freqmeas.register_parameter(osc.periodogram)

osc.dt(0.01)

with freqmeas.run() as datasaver:
    datasaver.add_result((osc.periodogram, osc.periodogram.get()))
    
dataid = datasaver.dataset
```

```{code-cell} ipython3
axs, cbax = plot_dataset(dataset)
aa = axs[0]
aa.set_yscale('log')
```

Just for the fun of it, let's make a measurement with the averaged periodogram.

```{code-cell} ipython3
no_of_avgs = 100

with freqmeas.run() as datasaver:
    
    temp_per = osc.periodogram()
    
    for _ in range(no_of_avgs-1):
        temp_per += osc.periodogram()
        
    datasaver.add_result((osc.periodogram, temp_per/no_of_avgs),
                         (osc.freq_axis, osc.freq_axis.get()))

dataset = datasaver.dataset
```

```{code-cell} ipython3
axs, cbax = plot_dataset(dataset)
aa = axs[0]
aa.set_yscale('log')
```

## Measurement 3: 2D Sweeping

```{code-cell} ipython3
meas = Measurement()
meas.register_parameter(osc.npts)
meas.register_parameter(osc.trace, setpoints=[osc.npts], paramtype='numeric')

with meas.run() as datasaver:

    osc.dt(0.001)
    
    for npts in [200, 400, 600, 800, 1000, 1200]:
        osc.npts(npts)
        datasaver.add_result((osc.trace, osc.trace.get()),
                             (osc.npts, osc.npts()))
        
dataset = datasaver.dataset
```

```{code-cell} ipython3
_ = plot_dataset(dataset)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
