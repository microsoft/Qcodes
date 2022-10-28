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

# QCoDeS Example with Signal Hound USB-SA124B ParameterWithSetpoints

+++

This example build on top of the example in [Qcodes example with Signal Hound USB-SA124B.ipynb](Qcodes-example-with-Signal-Hound-USB-SA124B.ipynb) and shows how this driver can be used with a ParameterWithSetpoints

```{code-cell} ipython3
import qcodes as qc
```

```{code-cell} ipython3
from qcodes.instrument_drivers.signal_hound.USB_SA124B import SignalHound_USB_SA124B
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.plotting import plot_by_id
```

```{code-cell} ipython3
mysa = SignalHound_USB_SA124B('mysa', dll_path='C:\\Program Files\\Signal Hound\\Spike\\sa_api.dll')
```

```{code-cell} ipython3
mysa.get_idn()
```

```{code-cell} ipython3
mysa.frequency(2e9)
mysa.span(0.5e6)
```

```{code-cell} ipython3
mysa.avg(1)
meas = Measurement()
meas.register_parameter(mysa.freq_sweep)
with meas.run() as datasaver:
    datasaver.add_result((mysa.frequency_axis, mysa.frequency_axis.get()),
                         (mysa.freq_sweep, mysa.freq_sweep.get(),))
    
    dataid = datasaver.run_id
plot_by_id(dataid)
```

```{code-cell} ipython3
mysa.avg(10)
meas = Measurement()
meas.register_parameter(mysa.freq_sweep)
with meas.run() as datasaver:
    datasaver.add_result((mysa.frequency_axis, mysa.frequency_axis.get()),
                         (mysa.freq_sweep, mysa.freq_sweep.get(),))
    
    dataid = datasaver.run_id
plot_by_id(dataid)
```

```{code-cell} ipython3
mysa.avg(100)
meas = Measurement()
meas.register_parameter(mysa.freq_sweep)
with meas.run() as datasaver:
    datasaver.add_result((mysa.frequency_axis, mysa.frequency_axis.get()),
                         (mysa.freq_sweep, mysa.freq_sweep.get(),))
    
    dataid = datasaver.run_id
plot_by_id(dataid)
```

```{code-cell} ipython3
mysa.frequency(3e9)
mysa.span(1e6)
meas = Measurement()
meas.register_parameter(mysa.freq_sweep)
with meas.run() as datasaver:
    datasaver.add_result((mysa.frequency_axis, mysa.frequency_axis.get()),
                         (mysa.freq_sweep, mysa.freq_sweep.get(),))
    dataid = datasaver.run_id
plot_by_id(dataid)
```

```{code-cell} ipython3
mysa.frequency(3e9)
mysa.span(1e8)
meas = Measurement()
meas.register_parameter(mysa.freq_sweep)
with meas.run() as datasaver:
    datasaver.add_result((mysa.frequency_axis, mysa.frequency_axis.get()),
                         (mysa.freq_sweep, mysa.freq_sweep.get(),))
    dataid = datasaver.run_id
plot_by_id(dataid)
```

```{code-cell} ipython3
mysa.close()
```
