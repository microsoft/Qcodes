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

# Qcodes example with Keysight N9030B

+++

This notebook shows how to use Keysight N9030B instrument driver in Spectrum Analyzer and Phase Noise Modes for Swept SA and Log Plot measurements respectively.

Let's begin!

```{code-cell} ipython3
import numpy as np
import os
from qcodes.dataset import initialise_or_create_database_at, load_or_create_experiment, Measurement
from qcodes.dataset.plotting import plot_dataset, plot_by_id

from qcodes.instrument_drivers.Keysight.N9030B import N9030B
```

```{code-cell} ipython3
driver = N9030B("n9030b","driver_address")
```

```{code-cell} ipython3
driver.IDN()
```

## Spectrum Analyzer mode with Swept SA measurement

```{code-cell} ipython3
sa = driver.sa
```

```{code-cell} ipython3
sa.setup_swept_sa_sweep(start=200, stop= 10e3, npts=20001)
```

### With QCoDeS Measurement

+++

Initialize database and begin experiment...

```{code-cell} ipython3
tutorial_db_path = os.path.join(os.getcwd(), 'tutorial.db')
initialise_or_create_database_at(tutorial_db_path)
load_or_create_experiment(experiment_name='tutorial_exp', sample_name="no sample")
```

```{code-cell} ipython3
meas1 = Measurement()
meas1.register_parameter(sa.trace)
```

```{code-cell} ipython3
with meas1.run() as datasaver:
    datasaver.add_result((sa.trace, sa.trace.get()))

dataset = datasaver.dataset
```

#### Plot data

```{code-cell} ipython3
_ = plot_dataset(dataset)
```

## Phase Noise mode with Log Plot measurement

```{code-cell} ipython3
pn = driver.pn
```

```{code-cell} ipython3
pn.setup_log_plot_sweep(start_offset=10, stop_offset=200, npts=1001)
```

### With QCoDeS Measurement

```{code-cell} ipython3
meas2 = Measurement()
meas2.register_parameter(pn.trace)
```

```{code-cell} ipython3
with meas2.run() as datasaver:
    datasaver.add_result((pn.trace, pn.trace.get()))

run_id = datasaver.run_id
```

#### Plot data

```{code-cell} ipython3
_ = plot_by_id(run_id)
```
