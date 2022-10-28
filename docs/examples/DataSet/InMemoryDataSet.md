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

# In memory dataset

+++

This notebooks explains an alternative way of measuring where the raw data is not written directly to a sqlite database file but only kept in memory with the ability to export the data after the measurement is completed. This may significantly speed up measurements where a lot of data is acquired but there is no protection against any data lose that may happen during a measurement. (Power loss, computer crash etc.) However, there may be situations where this trade-off is worthwhile. Please do only use the in memory dataset for measurements if you understand the risks. 

```{code-cell} ipython3
%matplotlib inline
import numpy.random as rd
import matplotlib.pyplot as plt
import numpy as np

import qcodes as qc
from qcodes.dataset import (
    load_or_create_experiment,
    load_by_guid,
    load_by_run_spec,
    initialise_or_create_database_at,
    Measurement,
    DataSetType,
)
from qcodes.tests.instrument_mocks import (
    DummyInstrument,
    DummyInstrumentWithMeasurement,
)
from qcodes.dataset.plotting import plot_dataset
```

Here we set up two mock instruments and a database to measure into:

```{code-cell} ipython3
# preparatory mocking of physical setup

dac = DummyInstrument("dac", gates=["ch1", "ch2"])
dmm = DummyInstrumentWithMeasurement(name="dmm", setter_instr=dac)

station = qc.Station(dmm, dac)
```

```{code-cell} ipython3
initialise_or_create_database_at("./in_mem_example.db")
exp = load_or_create_experiment(experiment_name="in_mem_exp", sample_name="no sample")
```

And run a standard experiment writing data to the database: 

```{code-cell} ipython3
meas = Measurement(exp=exp)
meas.register_parameter(dac.ch1)  # register the first independent parameter
meas.register_parameter(dmm.v1, setpoints=(dac.ch1,))  # now register the dependent oone

meas.write_period = 0.5

with meas.run() as datasaver:
    for set_v in np.linspace(0, 25, 10):
        dac.ch1.set(set_v)
        get_v = dmm.v1.get()
        datasaver.add_result((dac.ch1, set_v), (dmm.v1, get_v))

    dataset1D = datasaver.dataset  # convenient to have for data access and plotting
```

```{code-cell} ipython3
ax, cbax = plot_dataset(dataset1D)
```

The in memory measurement looks nearly identical with the only difference being that we explicitly pass in an Enum to select the dataset class that we want to use as a parameter to ``measurement.run``

The ``DataSetType`` Enum currently has 2 members representing the two different types of dataset supported.

```{code-cell} ipython3
with meas.run(dataset_class=DataSetType.DataSetInMem) as datasaver:
    for set_v in np.linspace(0, 25, 10):
        dac.ch1.set(set_v)
        get_v = dmm.v1.get()
        datasaver.add_result((dac.ch1, set_v), (dmm.v1, get_v))
    datasetinmem = datasaver.dataset
```

```{code-cell} ipython3
ax, cbax = plot_dataset(datasetinmem)
```

```{code-cell} ipython3
datasetinmem.run_id
```

When the measurement is performed in this way the data is not written to the database but the metadata (run_id, timestamps, snapshot etc.) is.

To preserve the raw data it must be exported it to another file format. See [Exporting QCoDes Datasets](./Exporting-data-to-other-file-formats.ipynb) for more information on exporting including how this can be done automatically.

```{code-cell} ipython3
datasetinmem.export("netcdf", path=".")
```

The `export_info` attribute contains information about locations where the file was exported to. We will use this below to show how the data may be reloaded from a netcdf file.

```{code-cell} ipython3
path_to_netcdf = datasetinmem.export_info.export_paths["nc"]
path_to_netcdf
```

As expected we can see this file in the current directory.

```{code-cell} ipython3
!dir
```

Note that you can interact with the dataset via the `cache` attribute of the dataset in the same way as you can with a regular dataset. However the in memory dataset does not implement methods that provide direct access to the data from the dataset object it self (get_parameter_data etc.) since these read data from the database. 

+++

## Reloading data from db and exported file

```{code-cell} ipython3
from qcodes import load_by_run_spec
from qcodes.dataset.plotting import plot_dataset
```

```{code-cell} ipython3
ds = load_by_run_spec(captured_run_id=datasetinmem.captured_run_id)
```

```{code-cell} ipython3
plot_dataset(ds)
```

When a dataset is loaded using ``load_by_run_spec`` and related functions QCoDeS will first check if the data is available in the database. If not if will check if the data has been exported to ``netcdf`` and then try to load the data from the last known export location. If this fails a warning will be raised and the dataset will be loaded without any raw data.

+++

A dataset can also be loaded directly from the netcdf file. See [Exporting QCoDes Datasets](./Exporting-data-to-other-file-formats.ipynb) for more information on how this is done. Including information about how you can change the ``netcdf`` location.
