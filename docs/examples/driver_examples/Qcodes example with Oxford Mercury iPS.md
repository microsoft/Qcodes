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

# QCoDeS Example with Mercury iPS

+++

## Initial instantiation/connection

```{code-cell} ipython3
from qcodes.instrument_drivers.oxford.MercuryiPS_VISA import MercuryiPS
from time import sleep
```

```{code-cell} ipython3
# Note that the MercuryiPS_VISA is a VISA instrument using
# a socket connection. The VISA resource name therefore
# contains the port number and the word 'SOCKET'
mips = MercuryiPS('mips', 'TCPIP0::192.168.15.106::7020::SOCKET')
```

## Basic driver idea

The driver mainly deals with **field values** in Tesla. The driver is aware of the field values in three coordinate systems, cartesian, spherical, and cylindrical. The driver thus exposes the field coordinates x, y, z, phi, theta, rho, and r. Each coordinate comes in two versions: **target** and **measured**.

The idea is that the magnetic field is always changed in two steps; first a target is set, then the magnet is asked to ramp to said target.

+++

## Safe regions

In addition to the safety limits baked in to the physical instrument, the driver can accept a safety limit function provided by the user. The function checks - upon receiving a new field target - whether the target is inside an allowed region.

The limit function must take input arguments Bx, By, Bz (in Tesla) and return a boolean that tells us whether that field value is safe.

```{code-cell} ipython3
# example: the safe region is a sphere
import numpy as np


def spherical_limit(x, y, z):
    """
    Safe region is a sphere of radius 1 T
    """
    return np.sqrt(x**2 + y**2 + z**2) <= 1

# assign the limit function (this can also be done at init)
mips.set_new_field_limits(spherical_limit)
```

## Two different ramps

The driver can perfom the ramp in two different ways: *simultaneous* ramping or *safe* ramping.

When simultaneously ramping, all three field components are ramped at the same time.
This method is non-blocking, and it is thus possible to query the field while it is ramping. The method does, however, **not** guarantee that the field stays inside the allowed region during the ramp. If the different axes have different ramp speeds, this is a real risk.

When safely ramping, all field components that are ramped *towards* the origin are ramped before those who are ramped *away from* the origin. The ramp is thus sequential and blocking, but if the safe region is convex (and contains the origin), you are guaranteed the the field never exceeds the safe region.

+++

## Parameter overview

```{code-cell} ipython3
mips.print_readable_snapshot(update=True)
```

## Ramp examples

+++

### First example: invalid targets

```{code-cell} ipython3
mips.x_target(1)  # so far, so good
try:
    mips.y_target(0.5)  # this takes us out of the unit sphere
except ValueError as e:
    print("Can not set that")
```

```{code-cell} ipython3
# reset and try in a different coordinate system
mips.x_target(0)
try:
    mips.r_target(1.1)
except ValueError as e:
    print("Can not set that")
```

### Second example: simul ramps to the origin

First we ramp the field to Bx = 1, By = 0, Bz = 0, then rotate out to thea=46, phi=30, then finally ramp it down to zero while measuring r, theta, and phi.

+++

#### STEP A

```{code-cell} ipython3
mips.GRPX.field_ramp_rate(0.01)
mips.GRPY.field_ramp_rate(0.01)
mips.GRPZ.field_ramp_rate(0.01)

mips.x_target(0.1)
mips.y_target(0)
mips.z_target(0)

mips.ramp(mode='simul')

# since simul mode is non-blocking,
# we can read out during the ramp
while mips.is_ramping():
    print(f'Ramping X to {mips.x_target()} T, now at {mips.x_measured()} T')
    sleep(1)
sleep(1)
print(f'Done ramping, now at {mips.x_measured()} T')
```

#### STEP B

Note that since the magnet itself has no notion of any other coordinate system than cartesian coordinates, it does **NOT** follow a path where r is constant. The user must **MANUALLY** ensure to break up a ramp where r is meant to be constant into sufficiently many small steps.

```{code-cell} ipython3
mips.theta_target(45)
mips.phi_target(30)
mips.r_target(0.1)

mips.ramp(mode='simul')

while mips.is_ramping():
    print(f"Ramping... r: {mips.r_measured():.6f} T, "
          f"theta: {mips.theta_measured():.2f}, "
          f"phi: {mips.phi_measured():.2f}")
    sleep(1)
print(f"Done... r: {mips.r_measured():.6f} T, "
      f"theta: {mips.theta_measured():.2f}, "
      f"phi: {mips.phi_measured():.2f}")
```

#### STEP C

```{code-cell} ipython3
mips.theta_target(45)
mips.phi_target(30)
mips.r_target(0)

mips.ramp(mode='simul')

# since simul mode is non-blocking,
# we can read out during the ramp
while mips.is_ramping():
    print(f"Ramping... r: {mips.r_measured():.6f} T, "
          f"theta: {mips.theta_measured():.2f}, "
          f"phi: {mips.phi_measured():.2f}")
    sleep(1)
print(f"Done... r: {mips.r_measured():.6f} T, "
      f"theta: {mips.theta_measured():.2f}, "
      f"phi: {mips.phi_measured():.2f}")
```

### Third example: safe ramp away from the origin

+++

At the origin, we can not meaningfully **measure** what theta and phi is, but the target values are persistent.

If we ramp up again and measure, we should thus get back to our target values. We use blocking safe ramp for this (just to also test/show a blocking ramp).

```{code-cell} ipython3
mips.r_target(0.05)

mips.ramp(mode='safe')

print('Ramped back out again.')
print(f'Field values are: theta: {mips.theta_measured()}, phi: {mips.phi_measured()}')
```

### That's it for now! Happy sweeping.

```{code-cell} ipython3
# sweep back down for good measures
mips.x_target(0)
mips.y_target(0)
mips.z_target(0)

mips.ramp(mode='safe')

mips.close()
```
