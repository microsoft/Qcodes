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

# Write data to cache

This notebook is meant to be used together with [Read data from cache](./read_data_from_cache.ipynb) to demonstate the use of the datasets cache.

+++

First we setup a simple experiment. This is copied from another notebook and can be ignored in this context.

```{code-cell} ipython3
%matplotlib notebook
import numpy.random as rd
import matplotlib.pyplot as plt
from functools import partial
import numpy as np

from time import sleep, monotonic

import qcodes as qc
from qcodes import Station, load_or_create_experiment, \
    initialise_database, Measurement, load_by_run_spec, load_by_guid
from qcodes.tests.instrument_mocks import DummyInstrument, DummyInstrumentWithMeasurement
from qcodes.dataset.plotting import plot_dataset
import time
```

```{code-cell} ipython3
# preparatory mocking of physical setup

dac = DummyInstrument('dac', gates=['ch1', 'ch2'])
dmm = DummyInstrumentWithMeasurement('dmm', setter_instr=dac)

station = qc.Station(dmm, dac)
```

```{code-cell} ipython3
initialise_database()
exp = load_or_create_experiment(experiment_name='dataset_cache_test',
                          sample_name="no sample")
```

Now we are ready to run an experiment. Once this experiment is running, take note of the id of the run (also accessible via ``dataset.captured_run_id``) created and open the [Read data from cache](./read_data_from_cache.ipynb) notebook and use there this id.  After 20 sec this notebook will start writing actual data to the dataset.

```{code-cell} ipython3


# And then run an experiment

meas = Measurement(exp=exp)
meas.register_parameter(dac.ch1)  # register the first independent parameter
meas.register_parameter(dmm.v1, setpoints=(dac.ch1,))  # now register the dependent oone

meas.write_period = 2


with meas.run() as datasaver:
    time.sleep(20)
    # While sleeping here start loader. From load_cached_notebook.ipynb
    # this is done by loading this new run via ``captured_run_id`` printed when the measurement starts
    print("done sleeping")
    for set_v in np.linspace(0, 25, 100):
        dac.ch1.set(set_v)
        get_v = dmm.v1.get()
        datasaver.add_result((dac.ch1, set_v),
                             (dmm.v1, get_v))
        # flush so this always works
        datasaver.flush_data_to_database(block=True)
        time.sleep(0.1)

    
    dataset = datasaver.dataset  # convenient to have for plotting
```

```{code-cell} ipython3

```
