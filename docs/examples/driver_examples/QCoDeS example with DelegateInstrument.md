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

# Qcodes example with DelegateInstrument driver

This notebooks explains how to use the `DelegateInstrument` driver.

## About
The goal of the `DelegateInstrument` driver is to make it easier to combine different parameters together into a new "virtual" instrument. Each parameter on a `DelegateInstrument` can point to one or more parameters on other instruments in the station.

## Usage
The way it's used is mainly by specifying an entry in the station YAML. For instance, let's say you want to use a magnetic field coil. The driver has a method set_field(value, block), that by default is set to block=True, which means the field is ramped in a way that blocks further execution until the desired value is reached. However, let's say you are creating a measurement in which you want the parameter to be set, and while the value is ramping, you want to measure other parameters. This can be done by using `DelegateInstrument` and specifying a custom `setter` for the parameter that gets and sets the magnetic field.

By default, each parameter is represented by a `DelegateParameter`. The `DelegateInstrument` also supports passing multiple source parameters to a given parameter. In order to do this, simply specify multiple parameters in the dictionary values  under the `parameters` key.

It can also add instrument channels, specified under a separate key `channels`, shown in the second half of the notebook.

```{code-cell} ipython3
%%writefile example.yaml

instruments:
  field_X:
    type: qcodes.tests.instrument_mocks.MockField

  field:
    type: qcodes.instrument.delegate.DelegateInstrument
    init:
      parameters:
        X:
          - field_X.field
        ramp_rate:
          - field_X.ramp_rate
        combined:
          - field_X.field
          - field_X.ramp_rate
      set_initial_values_on_load: true
      initial_values:
        ramp_rate: 1.0
      setters:
        X:
          method: field_X.set_field
          block: false
```

```{code-cell} ipython3
import qcodes as qc
```

```{code-cell} ipython3
station = qc.Station(config_file="example.yaml")
```

```{code-cell} ipython3
field_X = station.load_field_X()
field = station.load_field(station=station)
```

```{code-cell} ipython3
field.X()
```

```{code-cell} ipython3
field.X(1.)
```

```{code-cell} ipython3
field.X()
```

```{code-cell} ipython3
field.X()
```

```{code-cell} ipython3
field.X()
```

```{code-cell} ipython3
field.X()
```

As you can see, the field is now ramped in the background with the specified ramp rate. Now, let's try to create a measurement that uses this ability, and ramps the field in the background while measuring:

```{code-cell} ipython3
field.ramp_rate(10.)
field_X.field(0.0)
```

```{code-cell} ipython3
field.X()
```

```{code-cell} ipython3
import time
meas = qc.Measurement(station=station)
meas.register_parameter(field.X)

with meas.run() as datasaver:
    for B in [0.1, 0.0]:
        field.X(B)
        while field.X() != B:
            datasaver.add_result((field.X, field.X()))
            time.sleep(0.01)
    datasaver.flush_data_to_database()
```

```{code-cell} ipython3
datasaver.dataset.to_pandas_dataframe().plot()
```

When specifying multiple source parameters on a given parameter, the grouped parameter will automatically return a `namedtuple` that returns both values.

```{code-cell} ipython3
field.combined()
```

We can now also create a custom parameter that does a simple calculation based on the current parameters.

```{code-cell} ipython3
import numpy as np

def calculate_ramp_time(X, ramp_rate):
    """Calculate ramp time in seconds"""
    dfield = np.abs(field.target_field - X)
    return 60. * dfield/ramp_rate
```

```{code-cell} ipython3
field._create_and_add_parameter(
    group_name="ramp_time",
    station=station,
    paths=["field_X.field", "field_X.ramp_rate"],
    formatter=calculate_ramp_time
)
```

```{code-cell} ipython3
field.ramp_rate(1.0)
field.target_field = 0.1
field.ramp_time()
```

```{code-cell} ipython3
field.X(0.1)
```

```{code-cell} ipython3
field.ramp_time()
```

```{code-cell} ipython3
import time
time.sleep(1.)
field.ramp_time()
```

```{code-cell} ipython3
import time
time.sleep(1.)
field.ramp_time()
```

# Devices with channels

+++

The YAML file below specifies the instruments with the channels/parameters we wish to group into a new instrument, here called "device". The first example simply adds the channel 'as is' using self.add_submodule, while the readout parameter is added as a DelegateParameter.

```{code-cell} ipython3
%%writefile example.yaml

instruments:
  lockin:
    type: qcodes.tests.instrument_mocks.MockLockin

  dac:
    type: qcodes.tests.instrument_mocks.MockDAC

  device:
    type: qcodes.instrument.delegate.DelegateInstrument
    init:
      parameters:
        readout: lockin.X
      channels:
        gate_1: dac.ch01
      set_initial_values_on_load: true
      initial_values:
        readout: 1e-5
        gate_1.voltage.post_delay: 0.01
```

```{code-cell} ipython3
station = qc.Station(config_file="example.yaml")
```

```{code-cell} ipython3
lockin = station.load_lockin()
dac = station.load_dac()
device = station.load_device(station=station)
```

```{code-cell} ipython3
print(device.gate_1)
print(device.gate_1.voltage.post_delay)
```

```{code-cell} ipython3
print(device.gate_1.voltage())
device.gate_1.voltage(-0.6)
device.gate_1.voltage()
```

The second example adds a channel using a custom channel class, which takes the initial channel and its name as input and has a parameter current_valid_ranges.

```{code-cell} ipython3
%%writefile example.yaml

instruments:
  lockin:
    type: qcodes.tests.instrument_mocks.MockLockin

  dac:
    type: qcodes.tests.instrument_mocks.MockDAC

  device:
    type: qcodes.instrument.delegate.DelegateInstrument
    init:
      parameters:
        readout: lockin.X
      channels:
        type: qcodes.tests.instrument_mocks.MockCustomChannel
        gate_1:
          channel: dac.ch01
          current_valid_range: [-0.5, 0]
      set_initial_values_on_load: true
      initial_values:
        readout: 1e-5
```

```{code-cell} ipython3
lockin.close()
dac.close()
```

```{code-cell} ipython3
station = qc.Station(config_file="example.yaml")
lockin = station.load_lockin()
dac = station.load_dac()
```

```{code-cell} ipython3
device = station.load_device(station=station)
```

```{code-cell} ipython3
device.gate_1
```

```{code-cell} ipython3
device.gate_1.voltage(-0.3)
```

```{code-cell} ipython3
device.gate_1.voltage()
```

The MockCustomChannel has a parameter `current_valid_range`.

```{code-cell} ipython3
device.gate_1.current_valid_range()
```
