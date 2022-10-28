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

# QCoDeS Example with Signal Hound USB-SA124B

+++

Please also note the extensions to this driver in [Qcodes example with Signal Hound USB-SA124B ParameterWithSetpoints](Qcodes-example-with-Signal-Hound-USB-SA124B-ParameterWithSetpoints.ipynb)

```{code-cell} ipython3
%matplotlib notebook
```

```{code-cell} ipython3
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.plotting import plot_by_id
```

```{code-cell} ipython3
from qcodes.instrument_drivers.signal_hound.USB_SA124B import SignalHound_USB_SA124B
```

```{code-cell} ipython3
sh = SignalHound_USB_SA124B('mysignalhound')
```

## Frequency trace

+++

The primary functionality of the Signal Hound driver is to capture a frequency trace.
The frequency trace is defined by the center frequency and span. After changing any parameter on the Signal Hound
is is important to sync the parameters to the device or you will get a runtime error.

```{code-cell} ipython3
sh.frequency(1e9)
sh.span(10e6)
sh.configure()
```

We can now capture a trace.

```{code-cell} ipython3
meas = Measurement()
meas.register_parameter(sh.trace)

with meas.run() as datasaver:
    datasaver.add_result((sh.trace, sh.trace(),))
    runid = datasaver.run_id
```

```{code-cell} ipython3
plot_by_id(runid)
```

In this case we are not measuring any signal so as expected we see noise

+++

## Averaging

+++

The driver supports averaging over multiple traces simply by setting the number of averages

```{code-cell} ipython3
sh.avg(10)
```

```{code-cell} ipython3
meas = Measurement()
meas.register_parameter(sh.trace)

with meas.run() as datasaver:
    datasaver.add_result((sh.trace, sh.trace(),))
    runid = datasaver.run_id
```

```{code-cell} ipython3
plot_by_id(runid)
```

We note that the spread of the noise level has gone down compared to a single measurement

+++

## Power

+++

The Spectrum Analyzer also supports measuring the power at a specific frequency. This works by capturing a trace with a span of 250 KHz (The minimum supported)around the center frequency and returning the maximum value in that range.

```{code-cell} ipython3
sh.power()
```

```{code-cell} ipython3
sh.close()
```
