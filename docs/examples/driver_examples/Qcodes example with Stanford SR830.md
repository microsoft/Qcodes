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

# QCoDeS Example with Stanford SR830

+++

In this notebook, we are presenting how to connect to SR830 lock-in amplifier and read its buffer in QCoDeS. The instrument is not connected to any other instrument while running this notebook, so the buffer is only showing the instrument noise data.

+++

### Imports and connecting to the instrument

```{code-cell} ipython3
import qcodes as qc
import numpy as np
from time import sleep
from qcodes.instrument_drivers.stanford_research.SR830 import SR830
from qcodes.utils.dataset import doNd
from qcodes import load_or_create_experiment
from qcodes.instrument.base import Instrument
from qcodes.utils.validators import Numbers
```

```{code-cell} ipython3
sr = SR830('lockin', 'GPIB0::7::INSTR')
load_or_create_experiment(experiment_name='SR830_notebook')
```

Let's quickly look at the status of the instrument after connecting to it:

```{code-cell} ipython3
sr.print_readable_snapshot()
```

### Basics of reading values from the lockin

+++

A parameter say `complex_voltage` can be read from the lockin as follows.

```{code-cell} ipython3
sr.complex_voltage()
```

In fact, a method name `snap` is available on SR830 lockin which allows the user to read 2 to 6 parameters simultaneously out of the following.

```{code-cell} ipython3
from pprint import pprint
pprint(list(sr.SNAP_PARAMETERS.keys()))
```

Method `snap` can be used in the following manner.

```{code-cell} ipython3
sr.snap('x','y','phase')
```

### Changing the Sensitivity
The driver can change the sensitivity automatically according to the R value of the lock-in.
So instead of manually changing the sensitivity on the front panel, you can simply run this in your data acquisition or Measurement (max_changes is an integer defining how many steps the autorange can change the sensitivity, the default is 1 step):

```{code-cell} ipython3
sr.autorange(max_changes=2)
```

### Preparing for reading the buffer and measurement

+++

The SR830 has two internal data buffers corresponding to the displays of channel 1 and channel 2.
Here we present a simple way to use the buffers.
The buffer can be filled either at a constant sampling rate or by sending an trigger.
Each buffer can hold 16383 points. The buffers are filled simultaneously. The QCoDeS driver always pulls the entire buffer, so make sure to reset (clear) the buffer of old data before starting and acquisition.

We setup channel 1 and the buffer to be filled at a constant sampling rate:

```{code-cell} ipython3
sr.ch1_display('X')
sr.ch1_ratio('none')
sr.buffer_SR(512)  # Sample rate (Hz)
sr.buffer_trig_mode.set('OFF')
```

We fill the buffer for one second as shown below:

```{code-cell} ipython3
sr.buffer_reset()
sr.buffer_start() # Start filling the buffers with 512 pts/s
sleep(1)
sr.buffer_pause()  # Stop filling buffers
```

Now we run a QCoDeS Measurement using do0d to get the buffer and plot it:

```{code-cell} ipython3
doNd.do0d(sr.ch1_datatrace, do_plot=True)
```

### Software trigger
Below we will illustrate how a software trigger can be sent to fill the buffer on the instrument. For Illustrative purposes, we define a Dummy Generator, that we wish to set before each measurement.

```{code-cell} ipython3
class DummyGenerator(Instrument):

    def __init__(self, name, **kwargs):

        super().__init__(name, **kwargs)

        self.add_parameter('v_start',
                           initial_value=0,
                           unit='V',
                           label='v start',
                           vals=Numbers(0,1e3),
                           get_cmd=None,
                           set_cmd=None)

        self.add_parameter('v_stop',
                           initial_value=1,
                           unit='V',
                           label='v stop',
                           vals=Numbers(1,1e3),
                           get_cmd=None,
                           set_cmd=None)

        self.add_parameter('v_gen',
                           initial_value=0,
                           unit='V',
                           label='v_gen',
                           vals=Numbers(self.v_start(),self.v_stop()),
                           get_cmd=None,
                           set_cmd=None)
```

```{code-cell} ipython3
gen = DummyGenerator('gen')
```

We can now setup the lock-in to use the trigger

```{code-cell} ipython3
sr.ch1_ratio('none')
sr.buffer_SR("Trigger")
sr.buffer_trig_mode.set('ON')
```

We need to connect the data strored in the buffer to the correspointing values of the Dummy Generator,
i.e we need to give the setpoints for the data to be stored in the buffer.
For this purperse the driver has the convience function set_sweep_parameters, that generates the setpoint with units and labels corresponding to the independent parameter her (gen.v_gen).

```{code-cell} ipython3
sr.set_sweep_parameters(gen.v_gen,0,1,100)
```

To fill the buffer we iterate through values of the sweep_setpoints and change the value of the Dummy Generator followed by a software trigger. To get and plot the data we use the do0d.

```{code-cell} ipython3
sr.buffer_reset()
for v in sr.sweep_setpoints.get():
    gen.v_gen.set(v)
    sleep(0.04)
    sr.send_trigger()
```

```{code-cell} ipython3
doNd.do0d(sr.ch1_datatrace, do_plot=True)
```

We are not restricted to sample on an equally spaced grid. We can set the sweep_array directly.

```{code-cell} ipython3
grid_sample = np.concatenate((np.linspace(0, 0.5, 5),np.linspace(0.51, 1, 50)))
sr.sweep_setpoints.sweep_array = grid_sample
sr.sweep_setpoints.unit = gen.v_gen.unit
sr.sweep_setpoints.label = 'You also need to set label and unit'
```

```{code-cell} ipython3
sr.buffer_reset()
for v in grid_sample:
    gen.v_gen.set(v)
    sleep(0.04)
    sr.send_trigger()
```

```{code-cell} ipython3
doNd.do0d(sr.ch1_datatrace, do_plot=True)
```
