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

# QCoDeS Example with Keysight 33500B

```{code-cell} ipython3
import qcodes as qc
from qcodes.instrument_drivers.Keysight.KeysightAgilent_33XXX import WaveformGenerator_33XXX 
```

```{code-cell} ipython3
ks = WaveformGenerator_33XXX('ks', 'TCPIP0::K-33522B-00256::inst0::INSTR')
```

## Basic usage

```{code-cell} ipython3
# SET up a sawtooth
ks.ch1.function_type('RAMP')
ks.ch1.ramp_symmetry(100)
ks.ch1.amplitude_unit('VPP')
ks.ch1.amplitude(1)
ks.ch1.offset(0)
ks.ch1.frequency(2e3)
ks.sync.source(1)
```

```{code-cell} ipython3
# Start it
ks.sync.output('ON')
ks.ch1.output('ON')
```

```{code-cell} ipython3
ks.ch1.frequency(1e3)
```

```{code-cell} ipython3
# stop it
ks.sync.output('OFF')
ks.ch1.output('OFF')
```

## Using burst mode

+++

In burst mode, the instrument starts running a task (e.g. a waveform generation) upon receiving a trigger

```{code-cell} ipython3
# TRIGGERING

# Can be 'EXT' (external), 'IMM' (immediate, internal),
# 'BUS' (software trigger), 'TIM' (timed)
ks.ch1.trigger_source('EXT')  

ks.ch1.trigger_count(1)
ks.ch1.trigger_delay(0)  # seconds

# for external triggering, a slope should be specified
ks.ch1.trigger_slope('POS')

# For timed triggering, the time between each trigger should be set
ks.ch1.trigger_timer(50e-3)

# BURSTING

ks.ch1.burst_state('ON')
ks.ch1.burst_mode('N Cycle')  # Can be 'N Cycle' or 'Gated'

# when in 'N Cycle' mode, the following options are available
ks.ch1.burst_ncycles(1)  # Can be 1, 2, 3,... , 'MIN', 'MAX', or 'INF'
ks.ch1.burst_phase(180)  # the starting phase (degrees)

# If in 'Gated' mode, the following should be specified
ks.ch1.burst_polarity('NORM')  # Can be 'NORM' or 'INV'
```

## Error handling

```{code-cell} ipython3
# The instrument has an error queue of length up to 20 messages.
# The queue message retrieval is first-in-first-out

# The first (i.e. oldest) error message in the queue can be gotten (and thereby removed from the queue)
ks.error()
```

```{code-cell} ipython3
# The entire queue can be flushed out

# generate a few errors
for ii in range(3):
    ks.write('gimme an error!')

ks.flush_error_queue()
```

```{code-cell} ipython3

```
