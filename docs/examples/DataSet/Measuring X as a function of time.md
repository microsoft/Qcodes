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

# Measuring X as a function of time

Sometimes we'd like to measure something as a function of elapsed wall clock time. QCoDeS provides a convenient default way of doing such a measurement, namely by using the `ElapsedTimeParameter`.

The main utility of having a default way of measuring time is the uniformity in data of different experiments.

```{code-cell} ipython3
import os

import numpy as np

from qcodes.instrument.specialized_parameters import ElapsedTimeParameter
from qcodes.instrument.parameter import Parameter
from qcodes.dataset import initialise_or_create_database_at
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.experiment_container import load_or_create_experiment
from qcodes.dataset.plotting import plot_dataset
```

### Prepatory footwork: setup database and experiment

```{code-cell} ipython3
initialise_or_create_database_at(os.path.join(os.getcwd(), 'x_as_a_function_of_time.db'))
load_or_create_experiment('tutorial', 'no_sample')
```

## The measurement itself

We'll measure some Brownian motion. We set up a parameter for the noise.

```{code-cell} ipython3
noise = Parameter('noise',
                  label='Position',
                  unit='m',
                  get_cmd=lambda: np.random.randn())
time = ElapsedTimeParameter('time')
```

```{code-cell} ipython3
meas = Measurement()
meas.register_parameter(time)
meas.register_parameter(noise, setpoints=[time])
```

```{code-cell} ipython3
with meas.run() as datasaver:
    pos = 0
    time.reset_clock()
    for _ in range(100):
        pos += noise()
        now = time()
        datasaver.add_result((noise, pos), (time, now))

dataset = datasaver.dataset
```

```{code-cell} ipython3
axs, cbs = plot_dataset(dataset)
```
