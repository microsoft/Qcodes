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

# QCoDeS Example with the Rigol DS 1074 Z oscilloscope

+++

This notebook presents the implementation of QCoDeS driver for the Rigol DS1074Z oscilloscope.

+++

## Importing dependencies

```{code-cell} ipython3
#Qcodes import
from qcodes.dataset.plotting import plot_dataset
from qcodes.instrument_drivers.rigol.DS1074Z import DS1074Z
from qcodes import initialise_database, load_or_create_experiment, Measurement
```

## Create the instrument

+++

The instrument is created in the following way. The address can be found using the NI-MAX explorer.

```{code-cell} ipython3
rigol = DS1074Z('rigol', 'USB0::0x1AB1::0x04CE::DS1ZB161650342::INSTR')
```

## Trigger setup

+++

Trigger source can be set to any channel (1 to 4). Here we use the input signal from channel 1 itself as a source for trigger.

```{code-cell} ipython3
rigol.trigger_edge_source('ch1')
rigol.trigger_edge_slope('negative')
```

The trigger-mode type supported by this oscilloscope are `edge`, `pulse`, `video` and `pattern`. Both the trigger mode and trigger level can be set in the following manner.

```{code-cell} ipython3
rigol.trigger_mode('edge')
rigol.trigger_level(0.2)
```

## Data acquisition and plotting

+++

 This particular driver implements acquiring the trace as a `ParameterWithSetpoints`. Here, we show how the trace can be measured and plotted. For more information on `ParameterWithSetpoints` refer to [this notebook](http://qcodes.github.io/Qcodes/examples/Parameters/Simple-Example-of-ParameterWithSetpoints.html).

```{code-cell} ipython3
initialise_database()
exp = load_or_create_experiment(experiment_name='Oscilloscope trace',
                               sample_name='no_name')
```

```{code-cell} ipython3
meas_helper = Measurement(exp=exp)
meas_helper.register_parameter(rigol.channels.ch1.trace)

with meas_helper.run() as datasaver:
        datasaver.add_result((rigol.channels.ch1.trace, rigol.channels.ch1.trace()))
```

```{code-cell} ipython3
_ = plot_dataset(datasaver.dataset)
```
