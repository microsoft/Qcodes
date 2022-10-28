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

# QCoDeS Example with QDac_channels

```{code-cell} ipython3
import qcodes as qc
import numpy as np

from time import sleep

from qcodes.instrument_drivers.QDev.QDac_channels import QDac
```

```{code-cell} ipython3
# Connect to the instrument
qdac = QDac('qdac', 'ASRL6::INSTR', update_currents=False)
```

## Basic QDac Usage

The QCoDeS QDac driver currently supports using
  * 48 Output Channels
  * 3 $\times$ 6 temperature sensors

Each output channel has six parameters:
  * DC voltage
  * DC voltage range
  * Current out (read-only)
  * Current out range
  * slope
  * sync

The slope is the (maximal) slope in V/s that the channel can allow its voltage to change by. By default, all channels have a slope of "Inf". The slope can be changed dynamically, but no more than 8 channels can have a finite slope at any given time (this is due to hardware limitations).

```{code-cell} ipython3
# Current out is the current flowing through the channel this is read-only
print(qdac.ch01.i.get(), qdac.ch01.i.unit)
```

```{code-cell} ipython3
# The current range can be either 0 to 1 μA or 0 to 100 μA
print(qdac.ch01.irange.get())
# This is set with either 0 (0 to 1 μA) or 1 (0 to 100 μA) 
qdac.ch01.irange.set(1)
```

```{code-cell} ipython3
# The DC voltage may directly be set and gotten
qdac.ch01.v.set(-1)
print('Channel 1 voltage: {} {}'.format(qdac.ch01.v.get(), qdac.ch01.v.unit))
```

```{code-cell} ipython3
# Multiple channels can be addressed simultaneously via the 'channels' list
qdac.channels[0:20].v.get()
```

```{code-cell} ipython3
# Similarly, we may set them
qdac.channels[0:2].v.set(-1)
```

```{code-cell} ipython3
# The maximal voltage change (in V/s) may be set for each channel
qdac.ch01.slope.set(1)
qdac.ch02.slope.set(2)
# An overview may be printed (all other channels have 'Inf' slope)
qdac.printslopes()
```

```{code-cell} ipython3
# now setting channel 1 and 2 voltages will cause slow ramps to happen
qdac.ch01.v.set(0)
qdac.ch02.v.set(0)
```

```{code-cell} ipython3
# Note that only 8 (or fewer) channels can have finite slopes at one time
# To make space for other channels, set the slope to inifite
qdac.ch01.slope('Inf')
qdac.printslopes()
```

```{code-cell} ipython3
# To each channel one may assign a sync channel:
qdac.ch02.sync(2)  # sync output 2 will fire a 10 ms 5 V pulse when ch02 ramps
# note that even if no visible ramp is performed (i.e. ramping from 1 V to 1 V), a pulse is still fired.

# The sync pulse settings can be modified
qdac.ch02.sync_delay(0)  # The sync pulse delay (s)
qdac.ch02.sync_duration(25e-3)  # The sync pulse duration (s). Default is 10 ms.
```

```{code-cell} ipython3
qdac.ch02.v.set(0)
```

```{code-cell} ipython3
# syncs are unassigned by assigning sync 0
qdac.ch02.sync(0)
```

### Attention!

The v_range parameter is really controlling a 20 dB (amplitude factor 10) attenuator. Upon changing the vrange, the attenuator is **immediately** applied (or revoked). This will --irrespective of any slope set-- cause an instantaneous voltage change unless the channel voltage is zero. By default, all attenuators are off, and the voltage range is from -10 V to 10 V for all channels.

```{code-cell} ipython3
# Here is a small example showing what to look out for
#
qdac.ch01.vrange.set(0)  # Attenuation OFF (the default)
qdac.ch01.v.set(1.5)
qdac.ch01.vrange.set(1)  # Attenuation ON
print(qdac.ch01.v.get())  # Returns 0.15 V
qdac.ch01.v.set(0.1)
qdac.ch01.vrange.set(0)  # Attenuation OFF
print(qdac.ch01.v.get())  # returns 1 V
```

## Overview of channel settings

The driver provides a method for pretty-printing the state of all channels. On startup, all channels are queried for voltage and current across them, but the current query is very slow (blame the hardware).

The pretty-print method may or may not **update** the values for the currents, depending on the value of the `update_currents` flag. Each current reading takes some 200 ms, so updating all current values takes about 10 s.

```{code-cell} ipython3
qdac.print_overview(update_currents=True)
```

### Temperature sensors

Physically, the QDac consists of six boards each hosting eight channels. On three locations on each board, a temperature sensors is placed. These provide read-only parameters, named `tempX_Y` where `X` is the board number and `Y` the sensor number.

```{code-cell} ipython3
print(qdac.temp0_0.get(), qdac.temp0_0.unit)
print(qdac.temp2_1.get(), qdac.temp0_0.unit)
print(qdac.temp5_2.get(), qdac.temp0_0.unit)
```

```{code-cell} ipython3
qdac.close()
```
