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

# QCoDeS Example with Newport AG-UC8 Piezo Motion Controller

```{code-cell} ipython3
import time
import qcodes
from qcodes.instrument_drivers.Newport.AG_UC8 import Newport_AG_UC8
```

The driver communicates with the Newport AG-UC8 via an USB serial port. In our case the device appears on COM3. The corresponding VISA address is "ASRL3".

```{code-cell} ipython3
ctl = Newport_AG_UC8("Newport", "ASRL3")
```

```{code-cell} ipython3
ctl.get_idn()
```

```{code-cell} ipython3
ctl.reset()
```

In this example, an AG-M100L mount is connected to channel 1 of the controller. The mount can rotate about two axes. Let's first measure the current position on each axis. The position is returned as a number from 0 to 1000 corresponding to the full travel range of the mount.

Note that these commands are slow (about 30 seconds on our setup).

```{code-cell} ipython3
ctl.channels[0].axis1.measure_position()
```

```{code-cell} ipython3
ctl.channels[0].axis2.measure_position()
```

Now reset the step counter on axis 1.

```{code-cell} ipython3
ctl.channels[0].axis1.zero_position()
print(ctl.channels[0].axis1.steps())
```

Then rotate the mount about axis 1 over 500 steps relative to its current position.
Note that the "steps" used here are not the same unit as the absolute position measured above.
A "step" here is simply one step of the piezo actuator and the step size depends on the amplitude parameter and on properties of the individual mount.

```{code-cell} ipython3
ctl.channels[0].axis1.move_rel(500)
print("Status:", ctl.channels[0].axis1.status())
print("Accumulated steps:", ctl.channels[0].axis1.steps())
time.sleep(1)
print("Status:", ctl.channels[0].axis1.status())
print("Accumulated steps:", ctl.channels[0].axis1.steps())
```

Notice how the command "move_rel" returns immediately while the actual motion is still in progress. After about 1 second, the motion has completed. If we then query the accumulated step count, we see that it corresponds to the relative movement we commanded.

Now let's go back in the other direction.

```{code-cell} ipython3
ctl.channels[0].axis1.move_rel(-300)
time.sleep(1)
print("Accumulated steps:", ctl.channels[0].axis1.steps())
```

The time between steps and the step amplitude can be changed, separately for each axis.
This will affect future "move_rel" commands for that axis.

```{code-cell} ipython3
print("Initial step delay:", ctl.channels[0].axis1.step_delay())
print("Initial step amplitude:", ctl.channels[0].axis1.step_amplitude_neg())
ctl.channels[0].axis1.step_delay(50)
ctl.channels[0].axis1.step_amplitude_neg(20)
print("New step delay:", ctl.channels[0].axis1.step_delay())
print("New step amplitude:", ctl.channels[0].axis1.step_amplitude_neg())
```

```{code-cell} ipython3
ctl.channels[0].axis1.move_rel(-200)
```

There is also a command to move to an absolute position. The target position is specified in the same units as returned by "measure_position": a number from 0 to 1000 representing the full travel range of the axis.

This command performs a position measurement before moving to the target position. This is done by sweeping the axis until it touches the limit switches. Therefore this command is slow. Also it only works with actuators that have built-in limit switches (such as the AG-M100L).

```{code-cell} ipython3
ctl.channels[0].axis1.move_abs(500)
```

The current status of the limit switches can also be queried directly:

```{code-cell} ipython3
ctl.channels[0].limit_status()
```
