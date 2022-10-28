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

# QCoDeS Example with the Rigol DG 1062 Instrument

```{code-cell} ipython3
import time 

from qcodes.instrument_drivers.rigol.DG1062 import DG1062
```

Instantiate the driver 

```{code-cell} ipython3
gd = DG1062("gd", "TCPIP0::169.254.187.99::INSTR")
```

## Basic usage

+++

Accessing the channels

```{code-cell} ipython3
gd.channels[0]
# Or...
gd.ch1
```

Trun the output for channel 1 to "on"

```{code-cell} ipython3
gd.channels[0].state(1)
# This is idential to 
gd.ch1.state(1)
```

With `apply` we can check which waveform is being generated now, for example on channel 1

```{code-cell} ipython3
gd.channels[0].current_waveform()
```

We can also change the waveform 

```{code-cell} ipython3
gd.channels[0].apply(waveform="SIN", freq=2000, ampl=0.5, offset=0.0, phase=0.0)
```

Change individual settings like so: 

```{code-cell} ipython3
gd.channels[0].offset(0.1)
```

This works for every setting, except waveform, which is read-only

```{code-cell} ipython3
gd.channels[0].waveform()
```

```{code-cell} ipython3
try: 
    gd.channels[0].waveform("SIN")
except NotImplementedError: 
    print("We cannot set a waveform like this ")
```

We can however do this: 

```{code-cell} ipython3
gd.channels[0].sin(freq=1E3, ampl=1.0, offset=0, phase=0)
```

To find out which arguments are applicable to a waveform: 

+++

Find out which waveforms are available

```{code-cell} ipython3
print(gd.waveforms)
```

## Setting the impedance 

```{code-cell} ipython3
gd.channels[1].impedance(50)
```

```{code-cell} ipython3
gd.channels[1].impedance()
```

```{code-cell} ipython3
gd.channels[1].impedance("HighZ") 
```

Alternatively, we can do 

```python
gd.channels[1].impedance("INF")
```

```{code-cell} ipython3
gd.channels[1].impedance()
```

## Sync commands 

```{code-cell} ipython3
gd.channels[0].sync()
```

```{code-cell} ipython3
gd.channels[0].sync("OFF") 
```

Alternativly we can do 

```python
gd.channels[0].sync(0) 
```

```{code-cell} ipython3
gd.channels[0].sync()
```

```{code-cell} ipython3
gd.channels[0].sync(1)
```

Alternativly we can do

```python
gd.channels[0].sync("ON")
```

```{code-cell} ipython3
gd.channels[0].sync()
```

## Burst commands 

+++

### Internally triggered burst 

```{code-cell} ipython3
# Interal triggering only works if the trigger source is manual 
gd.channels[0].burst.source("MAN")
```

```{code-cell} ipython3
# The number of cycles is infinite 
gd.channels[0].burst.mode("INF")
```

If we want a finite number of cycles: 

```python
gd.channels[0].burst.mode("TRIG")
gd.channels[0].burst.ncycles(10000)
```

Setting a period for each cycle: 

```python
gd.channels[0].burst.period(1E-3)
```

```{code-cell} ipython3
# Put channel 1 in burst mode 
gd.channels[0].burst.on(1)
# Turn on the channel. For some reason, if we turn on the channel 
# immediately after turning on the burst, we trigger immediately. 
time.sleep(0.1)
gd.channels[0].state(1)
```

```{code-cell} ipython3
# Finally, trigger the AWG
gd.channels[0].burst.trigger()
```

### extranally triggered burst 

```{code-cell} ipython3
gd.channels[0].burst.source("EXT")
```

### Setting the idle level

```{code-cell} ipython3
# Set the idle level to First PoinT
gd.channels[0].burst.idle("FPT")
```

```{code-cell} ipython3
# We can also give a number 
gd.channels[0].burst.idle(0)
```

```{code-cell} ipython3

```
