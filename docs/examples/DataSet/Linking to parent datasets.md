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

# Linking to parent datasets

When performing a measurement in QCoDeS, it is possible to annotate the outcome (the dataset) of that measurement as having one or more parent datasets. This is done by adding a link to each parent dataset. This notebook covers the mechanisms to do that by going through a few practical examples.

```{code-cell} ipython3
import os
import datetime

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

from qcodes.dataset.measurements import Measurement
from qcodes.dataset.plotting import plot_dataset
from qcodes.dataset.data_set import load_by_run_spec
from qcodes.dataset.sqlite.database import initialise_or_create_database_at
from qcodes.dataset.experiment_container import load_or_create_experiment
```

```{code-cell} ipython3
now = str(datetime.datetime.now())
tutorial_db_path = os.path.join(os.getcwd(), 'linking_datasets_tutorial.db')
initialise_or_create_database_at(tutorial_db_path)
load_or_create_experiment('tutorial ' + now, 'no sample')
```

## Example 1: Measuring and then fitting

Say we measure some raw data and subsequently do a curve fit to those data. We'd like to save the fit as a separate dataset that has a link to the dataset of the original data. This is achieved in two steps.

+++

### Step 1: measure raw data

```{code-cell} ipython3
meas = Measurement()
meas.register_custom_parameter(name='time',
                               label='Time', unit='s',
                               paramtype='array')
meas.register_custom_parameter(name='signal',
                               label='Signal', unit='V',
                               paramtype='array',
                               setpoints=['time'])

N = 500

with meas.run() as datasaver:
    time_data = np.linspace(0, 1, N)
    signal_data = np.sin(2*np.pi*time_data) + 0.25*np.random.randn(N)
    
    datasaver.add_result(('time', time_data), ('signal', signal_data))
dataset = datasaver.dataset
```

```{code-cell} ipython3
cbs, axs = plot_dataset(dataset)
```

### Step 2: Set up a fit "measurement"

We now load the raw data dataset, set up a new measurement for the fit, register the raw data as a parent and save a fit.

+++

As the very first step, we supply a model to fit to.

```{code-cell} ipython3
def fit_func(x, a, b):
    return a*np.sin(2*np.pi*x)+b
```

Next, we set up the fitting measurement.

```{code-cell} ipython3
raw_data = load_by_run_spec(captured_run_id=dataset.captured_run_id)

meas = Measurement()
meas.register_custom_parameter('fit_axis',
                               label='Fit axis', unit='t',
                               paramtype='array')
meas.register_custom_parameter('fit_curve',
                               label='Fitted curve', unit='V',
                               paramtype='array',
                               setpoints=['fit_axis'])
meas.register_custom_parameter('fit_param_a',
                               label='Fitted parameter amplitude',
                               unit='V')
meas.register_custom_parameter('fit_param_b',
                               label='Fitted parameter offset',
                               unit='V')
meas.register_parent(parent=raw_data, link_type="curve fit")
```

As we now run the measurement, the parent datasets become available via the datasaver. The datasets appear in the order they were registered.

```{code-cell} ipython3
with meas.run() as datasaver:
    raw = datasaver.parent_datasets[0]
    xdata = np.ravel(raw.get_parameter_data()['signal']['time'])
    ydata = np.ravel(raw.get_parameter_data()['signal']['signal'])
    
    popt, pcov = opt.curve_fit(fit_func, xdata, ydata, p0=[1, 1])
    
    fit_axis = xdata
    fit_curve = fit_func(fit_axis, *popt)
    
    datasaver.add_result(('fit_axis', fit_axis),
                         ('fit_curve', fit_curve),
                         ('fit_param_a', popt[0]),
                         ('fit_param_b', popt[1]))
    
fit_data = datasaver.dataset
```

```{code-cell} ipython3
cbs, axs = plot_dataset(fit_data)
```

And just for completeness, let us plot both datasets on top of each other.

```{code-cell} ipython3
fig, ax = plt.subplots(1)
cbs, axs = plot_dataset(raw_data, axes=ax, label='data')
cbs, axs = plot_dataset(fit_data, axes=ax, label='fit', linewidth=4)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Signal (V)')
plt.legend()
```
