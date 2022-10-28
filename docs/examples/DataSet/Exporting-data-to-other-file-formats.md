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

# Exporting QCoDes Datasets

+++

This notebook demonstrates how we can export QCoDeS datasets to other file formats. 

+++

## Setup

+++

First, we borrow an example from the measurement notebook to have some data to work with.

```{code-cell} ipython3
%matplotlib inline
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import qcodes as qc
from qcodes.dataset import (
    initialise_or_create_database_at,
    load_or_create_experiment,
    Measurement,
    load_by_run_spec,
    load_from_netcdf,
)
from qcodes.tests.instrument_mocks import (
    DummyInstrument,
    DummyInstrumentWithMeasurement,
)
from qcodes.dataset.plotting import plot_dataset

qc.logger.start_all_logging()
```

```{code-cell} ipython3
# preparatory mocking of physical setup
dac = DummyInstrument("dac", gates=["ch1", "ch2"])
dmm = DummyInstrumentWithMeasurement("dmm", setter_instr=dac)
station = qc.Station(dmm, dac)
```

```{code-cell} ipython3
initialise_or_create_database_at("./export_example.db")
exp = load_or_create_experiment(
    experiment_name="exporting_data", sample_name="no sample"
)
```

```{code-cell} ipython3
meas = Measurement(exp)
meas.register_parameter(dac.ch1)  # register the first independent parameter
meas.register_parameter(dac.ch2)  # register the second independent parameter
meas.register_parameter(
    dmm.v2, setpoints=(dac.ch1, dac.ch2)
)  # register the dependent one
```

We then perform two very basic measurements using dummy instruments.

```{code-cell} ipython3
# run a 2D sweep

with meas.run() as datasaver:

    for v1 in np.linspace(-1, 0, 200, endpoint=False):
        for v2 in np.linspace(-1, 1, 201):
            dac.ch1(v1)
            dac.ch2(v2)
            val = dmm.v2.get()
            datasaver.add_result((dac.ch1, v1), (dac.ch2, v2), (dmm.v2, val))

dataset1 = datasaver.dataset
```

```{code-cell} ipython3
# run a 2D sweep

with meas.run() as datasaver:
    for v1 in np.linspace(0, 1, 200, endpoint=False):
        for v2 in np.linspace(1, 2, 201):
            dac.ch1(v1)
            dac.ch2(v2)
            val = dmm.v2.get()
            datasaver.add_result((dac.ch1, v1), (dac.ch2, v2), (dmm.v2, val))

dataset2 = datasaver.dataset
```

## Exporting data manually

+++

The dataset can be exported using the `export` method. Currently exporting to netcdf and csv is supported.

```{code-cell} ipython3
dataset2.export("netcdf", path=".")
```

The `export_info` attribute contains information about where the dataset has been exported to:

```{code-cell} ipython3
dataset2.export_info
```

Looking at the signature of export we can see that in addition to the file format we can set the `prefix` and `path` to export to.

```{code-cell} ipython3
?dataset2.export
```

## Export data automatically

+++

Datasets may also be exported automatically using the configuration options given in dataset config section. 
Here you can toggle if a dataset should be exported automatically using the `export_automatic` option as well as set the default type, prefix, elements in the name, and path. See [the table here](https://qcodes.github.io/Qcodes/user/configuration.html) for the relevant configuration options.

For more information about how to configure QCoDeS datasets see [the page about configuration](https://qcodes.github.io/Qcodes/user/configuration.html)  in the QCoDeS docs.

+++

## Importing exported datasets into a new database

+++

The above dataset has been created in the following database

```{code-cell} ipython3
qc.config.core.db_location
```

Now lets imagine that we move the exported dataset to a different computer. To emulate this we will create a new database file and set it as the active database. 

```{code-cell} ipython3
initialise_or_create_database_at("./reimport_example.db")
```

```{code-cell} ipython3
qc.config.core.db_location
```

We can then reload the dataset from the netcdf file as a DataSetInMem. This is a class that closely matches the regular DataSet class however its metadata may or may not be written to a database file and its data is not written to a database file. See more in [
In memory dataset](./InMemoryDataSet.ipynb) . Concretely this means that the data captured in the dataset can be acceced via `dataset.cache.data` etc. and not via the methods directly on the dataset (`dataset.get_parameter_data` ...) 

Note that it is currently only possible to reload a dataset from a netcdf export and not from a csv export. This is due to the fact that a csv export only contains the raw data and not the metadata needed to recreate a dataset.

```{code-cell} ipython3
loaded_ds = load_from_netcdf(dataset2.export_info.export_paths["nc"])
```

```{code-cell} ipython3
type(loaded_ds)
```

However, we can still export the data to Pandasa and xarray.

```{code-cell} ipython3
loaded_ds.cache.to_xarray_dataset()
```

And plot it using `plot_dataset`.

```{code-cell} ipython3
plot_dataset(loaded_ds)
```

Note that the dataset will have the same `captured_run_id` and `captured_counter` as before:

```{code-cell} ipython3
captured_run_id = loaded_ds.captured_run_id
captured_run_id
```

But do note that the `run_id` and `counter` are in general not preserved since they represent the datasets number in a given db file. 

```{code-cell} ipython3
loaded_ds.run_id
```

A loaded datasets metadata can be written to the current db file and subsequently the dataset including metadata and raw data reloaded from the database and netcdf file.

```{code-cell} ipython3
loaded_ds.write_metadata_to_db()
```

Now that the metadata has been written to a database the dataset can be plotted with [plottr](https://github.com/toolsforexperiments/plottr/) like a regular dataset.

```{code-cell} ipython3
del loaded_ds
```

```{code-cell} ipython3
reloaded_ds = load_by_run_spec(captured_run_id=captured_run_id)
```

```{code-cell} ipython3
plot_dataset(reloaded_ds)
```

Note that loading a dataset from the database will also load the raw data into `dataset.cache` provided that the `netcdf` file is still in the location where file was when the metadata was written to the database. Load_by_runspec and related functions will load data into a regular `DataSet` provided that the data can be found in the database otherwise it will be loaded into a `DataSetInMem`


If the netcdf file cannot be found the dataset will load with a warning and the raw data will not be accessible from the dataset. 

If this happens because you have moved the location of a netcdf file you can use the method ``set_netcdf_location`` to set a new location for the the netcdf file in the dataset and database file. 
Here we demonstrate this by copying the netcdf file and changing the location using this method.

```{code-cell} ipython3
filepath = dataset2.export_info.export_paths["nc"]
new_file_path = str(Path(dataset2.export_info.export_paths["nc"]).parent / "somefile.nc")
new_file_path
```

```{code-cell} ipython3
shutil.copyfile(dataset2.export_info.export_paths["nc"], new_file_path)
```

```{code-cell} ipython3
reloaded_ds.set_netcdf_location(new_file_path)
reloaded_ds.export_info
```
