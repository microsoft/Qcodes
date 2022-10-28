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

# QCoDeS Example with Lakeshore 325

Here provided is an example session with model 325 of the Lakeshore temperature controller

```{code-cell} ipython3
%matplotlib notebook
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

from qcodes.instrument_drivers.Lakeshore.Model_325 import Model_325
```

```{code-cell} ipython3
lake = Model_325("lake", "GPIB0::12::INSTR")
```

## Sensor commands

```{code-cell} ipython3
# Check that the sensor is in the correct status
lake.sensor_A.status()
```

```{code-cell} ipython3
# What temperature is it reading?
lake.sensor_A.temperature()
```

```{code-cell} ipython3
lake.sensor_A.temperature.unit
```

```{code-cell} ipython3
# We can access the sensor objects through the sensor list as well
assert lake.sensor_A is lake.sensor[0]
```

## Heater commands

```{code-cell} ipython3
# In a closed loop configuration, heater 1 reads from...
lake.heater_1.input_channel()
```

```{code-cell} ipython3
lake.heater_1.unit()
```

```{code-cell} ipython3
# Get the PID values
print("P = ", lake.heater_1.P())
print("I = ", lake.heater_1.I())
print("D = ", lake.heater_1.D())
```

```{code-cell} ipython3
# Is the heater on?
lake.heater_1.output_range()
```

## Loading and updating sensor calibration values

```{code-cell} ipython3
curve = lake.sensor_A.curve
```

```{code-cell} ipython3
curve_data = curve.get_data()
```

```{code-cell} ipython3
curve_data.keys()
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(curve_data["Temperature (K)"], curve_data['log Ohm'], '.')
plt.show()
```

```{code-cell} ipython3
curve.curve_name()
```

```{code-cell} ipython3
curve_x = lake.curve[23]
```

```{code-cell} ipython3
curve_x_data = curve_x.get_data()
```

```{code-cell} ipython3
curve_x_data.keys()
```

```{code-cell} ipython3
temp = np.linspace(0, 100, 200)
new_data = {"Temperature (K)": temp, "log Ohm": 1/(temp+1)+2}

fig, ax = plt.subplots()
ax.plot(new_data["Temperature (K)"], new_data["log Ohm"], '.')
plt.show()
```

```{code-cell} ipython3
curve_x.format("log Ohm/K")
curve_x.set_data(new_data)
```

```{code-cell} ipython3
curve_x.format()
```

```{code-cell} ipython3
curve_x_data = curve_x.get_data()
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(curve_x_data["Temperature (K)"], curve_x_data['log Ohm'], '.')
plt.show()
```

## Go to a set point

```{code-cell} ipython3
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
    ax.set_ylabel(f'Temperature, {channel_to_read.temperature.unit}')
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
lake.heater_1.control_mode("Manual PID")
lake.heater_1.output_range("Low (2.5W)")
lake.heater_1.input_channel("A")
# The following seem to be good settings for our setup
lake.heater_1.P(400)
lake.heater_1.I(40)
lake.heater_1.D(10)


lake.heater_1.setpoint(15.0)  # <- temperature
live_plot_temperature_reading(lake.sensor_a, n_reads=400)
```

## Querying the resistance and heater output

```{code-cell} ipython3
# to get the resistance of the system (25 or 50 Ohm)
lake.heater_1.resistance()
```

```{code-cell} ipython3
# to set the resistance of the system (25 or 50 Ohm)
lake.heater_1.resistance(50)
lake.heater_1.resistance()
```

```{code-cell} ipython3
# output in percent (%) of current or power, depending on setting, which can be queried by lake.heater_1.output_metric()
lake.heater_1.heater_output() # in %, 50 means 50%
```
