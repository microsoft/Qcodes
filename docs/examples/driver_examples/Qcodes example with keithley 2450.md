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

# QCoDeS Example with Tektronix Keithley 2450 Source Meter

In this example we will setup a number of [four-wire measurements](https://en.wikipedia.org/wiki/Four-terminal_sensing) with the 2540 source meter. We attach a variable resistor to the front terminals and determine if we can measure the correct resistance.

```{code-cell} ipython3
import qcodes as qc 
from qcodes.instrument_drivers.tektronix.Keithley_2450 import Keithley2450
from qcodes.dataset import initialise_database, Measurement, new_experiment
from qcodes.dataset.plotting import plot_dataset
from visa import VisaIOError
```

```{code-cell} ipython3
keithley = Keithley2450("keithley", "GPIB0::18::INSTR")
```

```{code-cell} ipython3
keithley.reset()
```

## Single point measurements 

+++

Attach a variable resistor to the front and source a current 

```{code-cell} ipython3
keithley.terminals("front")
keithley.source.function("current")
keithley.source.current(1E-6)  # Put 1uA through the resistor 
current_setpoint = keithley.source.current()

voltage = keithley.sense.function("voltage")
with keithley.output_enabled.set_to(True):
    voltage = keithley.sense.voltage()

print(f"Approx. resistance: ",  voltage/current_setpoint)
```

We can also directly measure the resistance

```{code-cell} ipython3
voltage = keithley.sense.function("resistance")
with keithley.output_enabled.set_to(True):
    resistance = keithley.sense.resistance()

print(f"Measured resistance: ",  resistance)
```

In 'current' mode, we cannot set/get a voltage and vice versa

```{code-cell} ipython3
try: 
    keithley.source.voltage()
except AttributeError as err: 
    function = keithley.source.function()
    print(f"In the '{function}' source mode the source module does not have a 'voltage' attribute")
```

This goes for both the source and sense subsystems 

```{code-cell} ipython3
try: 
    keithley.sense.current()
except AttributeError as err:
    function = keithley.sense.function()
    print(f"In the '{function}' sense mode the sense module does not have a 'current' attribute")
```

We also need to make sure the output is enabled for use the measure (or 'sense') a current or voltage 

+++

## Sweeping measurements 

+++

The instrument has a build-in sweep system. For the first measurement, we drive a current through the resistor and measure the voltage accross it. 

```{code-cell} ipython3
initialise_database()
experiment = new_experiment(name='Keithley_2450_example', sample_name="no sample")
```

Sweep the current from 0 to 1uA in 10 steps and measure voltage

```{code-cell} ipython3
keithley.sense.function("voltage")
keithley.sense.auto_range(True)

keithley.source.function("current")
keithley.source.auto_range(True)
keithley.source.limit(2)
keithley.source.sweep_setup(0, 1E-6, 10)

keithley.sense.four_wire_measurement(True)
```

```{code-cell} ipython3
meas = Measurement(exp=experiment)
meas.register_parameter(keithley.sense.sweep)

with meas.run() as datasaver:
    datasaver.add_result((keithley.source.sweep_axis, keithley.source.sweep_axis()),
                         (keithley.sense.sweep, keithley.sense.sweep()))

    dataid = datasaver.run_id

plot_dataset(datasaver.dataset)
```

Sweep the voltage from 10mV in 10 steps and measure current 

```{code-cell} ipython3
keithley.sense.function("current")
keithley.sense.range(1E-5)
keithley.sense.four_wire_measurement(True)

keithley.source.function("voltage")
keithley.source.range(0.2)
keithley.source.sweep_setup(0, 0.01, 10)
```

```{code-cell} ipython3
meas = Measurement(exp=experiment)
meas.register_parameter(keithley.sense.sweep)

with meas.run() as datasaver:
    datasaver.add_result((keithley.source.sweep_axis, keithley.source.sweep_axis()),
                         (keithley.sense.sweep, keithley.sense.sweep()))

    dataid = datasaver.run_id

plot_dataset(datasaver.dataset)
```

## To perform measurements with user-defined reading buffer

```{code-cell} ipython3
keithley.reset()
```

By default, when performing measurement, the value is stored in the default buffer "defbuffer1".

```{code-cell} ipython3
keithley.sense_function('current')
with keithley.output_enabled.set_to(True):
    data_point01 = keithley.sense.current()
    data_point02 = keithley.sense.current()
    data_point03 = keithley.sense.current()
print(f"The current measurements are {data_point01}, {data_point02}, {data_point03} A.")
```

We can use a user-defined reading buffer for measurement. The following example is to do a sweep measurement, and read extra data elements in addition to the measurement value with the new method "elements".

```{code-cell} ipython3
buffer_name = 'userbuff1'
buffer_size = 100
with keithley.buffer(buffer_name, buffer_size) as buff1:
    buff1.elements(['time', 'date', 'measurement', 'source_value_formatted'])
    keithley.source.sweep_setup(0, 1E-6, 10, buffer_name=buff1.buffer_name)
    data = keithley.sense.sweep()
    all_data = keithley.sense.sweep.get_selected()
```

"data" includes the numerical value of the measurement:

```{code-cell} ipython3
data
```

"all_data" includes extra information specified by the "elements()" method:

```{code-cell} ipython3
all_data
```

By using "with ... as ...:" to perform the measurement, there user-defined buffer is automatically removed after the measurement. **This is the recommanded way to use the user-defined buffer.**

```{code-cell} ipython3
try:
    buff1.size()
except VisaIOError as err:
    print(err)
```

And we can still access the data in the default buffer:

```{code-cell} ipython3
buffer_name = 'defbuffer1'
buffer = keithley.buffer(buffer_name)
```

```{code-cell} ipython3
print(f"There are {buffer.number_of_readings()} data points in '{buffer_name}'.")
```

The last reading is:

```{code-cell} ipython3
buffer.get_last_reading()
```

We can get all 3 previously measured data as following:

```{code-cell} ipython3
buffer.get_data(1,3)
```

And the original infomration are still there:

```{code-cell} ipython3
buffer.elements(["time", "measurement_formatted"])
buffer.get_data(1, 3)
```

This is all the available elements, if none is requested, "measurement" will be used:

```{code-cell} ipython3
buffer.available_elements
```

If the user gives a incorrect element name, error message will show which ones are correct:

```{code-cell} ipython3
try:
    buffer.elements(['dates'])
except ValueError as err:
    print(err)
    
```
