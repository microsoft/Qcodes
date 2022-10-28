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

# QCoDeS Example with the Lakeshore Model 372 to Control the Temperature of the Bluefors Fridge

The Lakeshore Temperature Controller Model 372 is used to control the temperature of the Bluefors fridges.

To use it as such outside of the control software provided by Bluefors, one has to establish an addtional connection. Within the Bluefors system, the Lakeshore is connected via its usb port (through a USB hub along with the other devices) to the control Laptop (as part of the Bluefors setup). To control the temperature of the fridge via QCoDeS, it is the most convenient to connect the Lakeshore via its LAN port to the control computer (the one with QCoDeS, not the one from Bluefors). In order to reach the LAN port of the Lakeshore, the Bluefors rack has to be opened, and the PCB board that is fixed to the metal board has to be opened as well (it's a door as well with magnetic clips on one side). Do NOT disconnect the USB! Then switch the operation mode (usually there is an Interface button on the instrument) from USB to LAB. When using a router, remember to set the IP address setting to DHCP. Finally, use the following address format for VISA address: `"TCPIP::<ip address>::<port>::SOCKET"`, where "port" is a known value from the manual (most probably, "7777").

As mentioned above, for using the Lakeshore with QCoDeS, the operation mode has to be switched from USB to LAN. When done with the measurements, please, switch back to USB, so that the logging of the Temperature provided by the Bluefors software continues. It is planned to implement a server that takes care of the logging, so that the switching to USB will no longer be necessary. For the time being, please, always remember to switch back!

+++

## Driver Setup

This notebook is using a simulated version of the driver, so that it can be run and played with, without an actual instrument. When trying it out with a real Lakeshore, please set `simulation = False`.

```{code-cell} ipython3
simulation = True
```

```{code-cell} ipython3
if simulation:
    from qcodes.tests.drivers.test_lakeshore import Model_372_Mock as Model_372
    import qcodes.instrument.sims as sims
    visalib = sims.__file__.replace('__init__.py',
                                    'lakeshore_model372.yaml@sim')
    ls = Model_372('lakeshore_372', 'GPIB::3::65535::INSTR',
                    visalib=visalib, device_clear=False)
else:
    from qcodes.instrument_drivers.Lakeshore.Model_372 import Model_372
    #                               put visa address here, see e.g. NI Max
    #                               or look up the IP address on
    #                               the instrument itself
    ls = Model_372('lakeshore_372', 'TCPIP::192.168.0.160::7777::SOCKET')
```

## Readout Sensor Channels

The lakeshore has two types of *channels*: *Readout channels* and *heaters*. For reading the temperature we use the readout channels. There are sixteen channels, each of which has the following parameters:

```{code-cell} ipython3
ls.ch01.parameters
```

All the parameters have docstrings, labels, and units, when applicable.

Some of these parameters have been added just because other interesting parameters can only be set together with these (Lakeshore uses VISA commands with multiple inputs/outputs).

Some parameters like `curve_number` should not be changed, unless the user knows what he's doing.

+++

In order to read temperature values from all the sensors, we can do the following:

```{code-cell} ipython3
for ch in ls.channels:
    print(f'Temperature of {ch.short_name} ({"on" if ch.enabled() else "off"}): {ch.temperature()} {ch.units()}')
```

The `enabled` parameter of the sensor channel is very important because Lakeshore gets readings from the enabled channels in sequence. This means that if you have 3 channels enabled, while you are contantly requesting the temperature reading from only the first one, the array of readings will have parts when the value is constant. This is because within those parts Lakeshore was busy with reading temperature from the other two channels.

The `units` parameter is also of big importance. As it will be explained below, it defines the units from the `setpoint` value of the heater that is used in a `closed_loop` mode.

+++

## Heating and feedback loop

To set a certain temperature one needs to start a feedback loop that reads the temperature from a sensor channel, and feeds it back to the sample through a heater. The Lakeshore 372 has three heaters: `sample_heater`, `warmup_heater`, and `analog_heater`.

Here the `sample_heater` will be used. It has the following parameters:

```{code-cell} ipython3
h = ls.sample_heater
h.parameters
```

The allowed modes, polarities, and ranges are defined in:

```{code-cell} ipython3
h.MODES
```

```{code-cell} ipython3
h.RANGES
```

```{code-cell} ipython3
h.POLARITIES
```

### Working with closed loop control

+++

To use a closed loop control, we need to set the `P`, `I`, `D` values, choose an `input_channel` that will be read within the closed loop, set the range of the heater (`output_range`), set the `setpoint` value (e.g. the target temperature), and start the operation by setting `mode` to `closed_loop`.

```{code-cell} ipython3
h.P(10)
h.I(10)
h.D(0)
h.output_range('31.6μA')
h.input_channel(9)
```

```{code-cell} ipython3
h.setpoint(0.01)
h.mode('closed_loop')
```

#### Units of the setpoint

+++

Be careful when setting the value of the `setpoint` - Lakeshore uses "preferred units" for it which are determined by the `units` parameter of the chosen `input_channel`. Thanks to that, Lakeshore 372 supports setting `setpoint` in `ohms` and `kelvins`.

```{code-cell} ipython3
ls.ch09.units()
```

```{code-cell} ipython3
print(h.setpoint.__doc__)  # when working in Jupyter, just use `h.setpoint?` syntax
```

#### Disable unrelated channels for continuos readings

+++

Note that in order to have Lakeshore constantly reading from the `input_channel`, you need to disable other channels. Otherwise, Lakeshore will be reading all the enabled channels one by one, which will slow down the convergence of the control loop.

```{code-cell} ipython3
ls.ch03.enabled(False)
```

#### Observe control loop working

+++

Now we can observe how the temperature gets steered towards the setpoint (This is not implemented in the simulated instrument)

```{code-cell} ipython3
import time
for i in range(5):
    time.sleep(0.1)
    print(f'T = {ls.ch09.temperature()}')
```

Textual representation is not very convenient, hence let's do the same but now with plotting (This is not implemented in the simulated instrument):

```{code-cell} ipython3
%matplotlib notebook

import time
import numpy
from IPython.display import display
from ipywidgets import interact, widgets
from matplotlib import pyplot as plt

def live_plot_temperature_reading(channel_to_read, read_period=0.2, n_reads=1000):
    """
    Live plot the temperature reading from a Lakeshore sensor channel

    Args:
        channel_to_read
            Lakeshore channel object to read the temperature from
        read_period
            time in seconds between two reads of the temperature
        n_reads
            total number of reads to perform
    """

    # Make a widget for a text display that is contantly being updated
    text = widgets.Text()
    display(text)

    fig, ax = plt.subplots(1)
    line, = ax.plot([], [], '*-')
    ax.set_xlabel('Time, s')
    ax.set_ylabel(f'Temperature, {channel_to_read.units()}')
    fig.show()
    plt.ion()

    for i in range(n_reads):
        time.sleep(read_period)

        # Update the text field
        text.value = f'T = {channel_to_read.temperature()}'

        # Add new point to the data that is being plotted
        line.set_ydata(numpy.append(line.get_ydata(), channel_to_read.temperature()))
        line.set_xdata(numpy.arange(0, len(line.get_ydata()), 1)*read_period)

        ax.relim()  # Recalculate limits
        ax.autoscale_view(True, True, True)  # Autoscale
        fig.canvas.draw()  # Redraw
```

```{code-cell} ipython3
live_plot_temperature_reading(channel_to_read=ls.ch09, read_period=0.01, n_reads=5)
```

## Waiting to reach setpoint
As we have seen, the call of the parameter `setpoint` is non-blocking. That means the function returns imediately without waiting for the setpoint to be reached. In many use-cases it is desirable to wait until a certain temperature regime has been reached. This can be achieved with `wait_until_set_point_reached` method. There are also three parameters which allow to tune the behavior of this method.

See below:

```{code-cell} ipython3
# time before reading the next temperature value
# in other words, wait half a second, then read the temperature and compare to setpoint
h.wait_cycle_time(0.5)

# wait until temperature within 5% of the setpoint
# the tolerance is defined as: |t_reading-t_setpoint|/t_reading
h.wait_tolerance(0.05)

# wait until temperature has been within the tolerance zone
# for `wait_equilibration_time` seconds
h.wait_equilibration_time(1.5)
```

```{code-cell} ipython3
# do the waiting:
if not simulation:  # does not work with simulated instrument!
    h.wait_until_set_point_reached()
```

For the simulation purposes, we can fake the heating of the sample by calling the `start_heating` method which only exists for the simulated instrument.

```{code-cell} ipython3
if simulation:
    ls.sample_heater.setpoint(4)
    ls.start_heating()  # starts from 7K and goes down to the setpoint of 4K
    ls.sample_heater.wait_until_set_point_reached()
```

## Automatically selecting a heater range (based on temperature)
To automatically select a heater range, one can define temperature limits for the individual heater ranges:

```{code-cell} ipython3
# all limits in K, 8 limits starting with limit for 31.6μA range
h.range_limits([0.021, 0.1, 0.2, 1.1, 2, 4, 8, 16])
```

```{code-cell} ipython3
list(h.RANGES.keys())
```

This means: from `0 K` to `0.021 K` use `31.6μA`, from `0.021 K` to `0.1 K` use `100μA`, and so on.

We can now set the range by giving a temperature using the `set_range_from_temperature` method:

```{code-cell} ipython3
h.set_range_from_temperature(0.15)
h.output_range()
```

## Sweeping/blocking paramameter
For compatibility with the legacy Loop construct, the Lakeshore driver exposes a blocking temperature parameter: `blocking_t`.
The setter for this parameter simply does:

+++

```python
def _set_blocking_t(self, t):
     self.set_range_from_temperature(t)
     self.setpoint(t)
     self.wait_until_set_point_reached()
```

+++

This parameter can be used in a `doNd`-like loop.

Note that the range only gets set at the beginning of the sweep, i.e. according to the setpoint not according to the temperature reading.
