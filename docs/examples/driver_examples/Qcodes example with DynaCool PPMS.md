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

# QCoDeS Example with DynaCool PPMS

This notebook explains how to control the DynaCool PPMS from QCoDeS.

For this setup to work, the proprietary `PPMS Dynacool` application (or, alternatively `Simulate PPMS Dynacool`) must be running on some PC. On that same PC, the `server.py` script (found in `qcodes/instrument_drivers/QuantumDesign/DynaCoolPPMS/private`) must be running. The script can be run from the command line with no arguments and will run under python 3.6+.

The architecture is as follows:

The QCoDeS driver sends strings via VISA to the server who passes those same strings on to the `CommandHandler` (found in `qcodes/instrument_drivers/QuantumDesign/DynaCoolPPMS/commandhandler`). The `CommandHandler` makes the calls into the proprietary API. The QCoDeS driver can thus be called on any machine that can communicate with the machine hosting the server.

Apart from that, the driver is really simple. For this notebook, we used the `Simulate PPMS Dynacool` application running on the same machine as QCoDeS.

```{code-cell} ipython3
%matplotlib notebook
from qcodes.instrument_drivers.QuantumDesign.DynaCoolPPMS.DynaCool import DynaCool
```

To instantiate the driver, simply provide the address and port in the standard VISA format.
The connect message is not too pretty, but there does not seem to be a way to query serial and firmware versions.

```{code-cell} ipython3
dynacool = DynaCool('dynacool', address="TCPIP0::127.0.0.1::5000::SOCKET")
```

To get an overview over all available parameters, use `print_readable_snapshot`.

A value of "Not available" means (for this driver) that the parameter has been deprecated.

```{code-cell} ipython3
dynacool.print_readable_snapshot(update=True)
```

## Temperature Control

As soon as ANY of the temperature rate, the temperature setpoint, or the temperature settling mode parameters has been set, the system will start moving to the given temperature setpoint at the given rate using the given settling mode.

The system can continuously be queried for its temperature.

```{code-cell} ipython3
from time import sleep
import matplotlib.pyplot as plt
import numpy as np

# example 1

dynacool.temperature_rate(0.1)
dynacool.temperature_setpoint(dynacool.temperature() - 1.3)

temps = []

while dynacool.temperature_state() == 'tracking':
    temp = dynacool.temperature()
    temps.append(temp)
    sleep(0.75)
    print(f'Temperature is now {temp} K')

```

```{code-cell} ipython3
plt.figure()
timeax = np.linspace(0, len(temps)*0.2, len(temps))
plt.plot(timeax, temps, '-o')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
```

## Field Control

The field has **five** related parameters:

- `field_measured`: The (read-only) field strength right now.
- `field_target`: The target field that the `ramp` method will ramp to when called. Setting this parameter does **not** trigger a ramp
- `field_rate`: The field ramp rate with initial value of `0`.
- `field_approach`: The approach that the system should use to ramp. By default it is set to `linear`.
- `field_ramp`: This is a convenience parameter that sets the target field and then triggers a blocking ramp.

The idea is that the user first sets the `field_target` and then ramps the field to that target using the `ramp` method. The ramp method takes a `mode` argument that controls whether the ramp is blocking or non-blocking. 

Using the simulation software, the field change is instanteneous irrespective of rate. We nevertheless include two examples of ramping here.

+++

### A blocking ramp

+++

First, we set a field target:

```{code-cell} ipython3
field_now = dynacool.field_measured()
target = field_now + 1
dynacool.field_target(target)
```

Note that the field has not changed yet:

```{code-cell} ipython3
assert dynacool.field_measured() == field_now
```

And now we ramp:

```{code-cell} ipython3
dynacool.ramp(mode='blocking')
```

The ramping will take some finite time on a real instrument. The field value is now at the target field:

```{code-cell} ipython3
print(f'Field value: {dynacool.field_measured()} T')
print(f'Field target: {dynacool.field_target()} T')
```

### A non-blocking ramp

The non-blocking ramp is very similar to the the blocking ramp.

```{code-cell} ipython3
field_now = dynacool.field_measured()
target = field_now - 0.5
dynacool.field_target(target)

assert dynacool.field_measured() == field_now

dynacool.ramp(mode='non-blocking')
# Here you can do stuff while the magnet ramps

print(f'Field value: {dynacool.field_measured()} T')
print(f'Field target: {dynacool.field_target()} T')
```

### Using the `field_ramp` parameter

The `field_ramp` parameter sets the target field and ramp when being set.

```{code-cell} ipython3
print(f'Now the field is {dynacool.field_measured()} T...')
print(f'...and the field target is {dynacool.field_target()} T.')
```

```{code-cell} ipython3
dynacool.field_ramp(1)
```

```{code-cell} ipython3
print(f'Now the field is {dynacool.field_measured()} T...')
print(f'...and the field target is {dynacool.field_target()} T.')
```

```{code-cell} ipython3

```
