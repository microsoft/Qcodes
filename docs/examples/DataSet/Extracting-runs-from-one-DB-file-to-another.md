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

# Extracting runs from one DB file to another

This notebook shows how to use the `extract_runs_into_db` function to extract runs from a database (DB) file (the source DB) into another DB file (the target DB). If the target DB does not exist, it will be created. The runs are **NOT** removed from the original DB file; they are copied over.

+++

## Setup

Let us set up a DB file with some runs in it.

```{code-cell} ipython3
import os

import numpy as np

from qcodes.dataset import extract_runs_into_db, load_or_create_experiment, load_experiment_by_name, Measurement
from qcodes.tests.instrument_mocks import DummyInstrument
from qcodes.station import Station

# The following function is imported and used here only for the sake
# of explicitness. As a qcodes user, please, consider this function
# private to qcodes which means its name, behavior, and location may
# change without notice between qcodes versions.
from qcodes.dataset.sqlite.database import connect
```

```{code-cell} ipython3
source_path = os.path.join(os.getcwd(), "extract_runs_notebook_source.db")
target_path = os.path.join(os.getcwd(), "extract_runs_notebook_target.db")
```

```{code-cell} ipython3
source_conn = connect(source_path)
target_conn = connect(target_path)
```

```{code-cell} ipython3
exp = load_or_create_experiment(
    experiment_name="extract_runs_experiment", sample_name="no_sample", conn=source_conn
)

my_inst = DummyInstrument("my_inst", gates=["voltage", "current"])
station = Station(my_inst)
```

```{code-cell} ipython3
meas = Measurement(exp=exp)
meas.register_parameter(my_inst.voltage)
meas.register_parameter(my_inst.current, setpoints=(my_inst.voltage,))

# Add 10 runs with gradually more and more data

for run_id in range(1, 11):
    with meas.run() as datasaver:
        for step, noise in enumerate(np.random.randn(run_id)):
            datasaver.add_result((my_inst.voltage, step), (my_inst.current, noise))
```

## Extraction

Now let us extract runs 3 and 7 into our desired target DB file. All runs must come from the same experiment. To extract runs from different experiments, one may call the function several times.

The function will look in the target DB to see if an experiment with matching attributes already exists. If not, such an experiment is created.

```{code-cell} ipython3
extract_runs_into_db(source_path, target_path, 3, 7)
```

```{code-cell} ipython3
target_exp = load_experiment_by_name(name="extract_runs_experiment", conn=target_conn)
```

```{code-cell} ipython3
target_exp
```

The last number printed in each line is the number of data points. As expected, we get 3 and 7.

Note that the runs will have different `run_id`s in the new database. Their GUIDs are, however, the same (as they must be).

```{code-cell} ipython3
exp.data_set(3).guid
```

```{code-cell} ipython3
target_exp.data_set(1).guid
```

Furthermore, note that the original `run_id` preserved as `captured_run_id`. We will demonstrate below how to look up data via the `captured_run_id`.

```{code-cell} ipython3
target_exp.data_set(1).captured_run_id
```

## Merging data from 2 databases

+++

There are occasions where it is convenient to combine data from several databases.

+++

Let's first demonstrate this by creating some new experiments in another db file.

```{code-cell} ipython3
extra_source_path = os.path.join(os.getcwd(), "extract_runs_notebook_source_aux.db")
```

```{code-cell} ipython3
source_extra_conn = connect(extra_source_path)
```

```{code-cell} ipython3
exp = load_or_create_experiment(
    experiment_name="extract_runs_experiment_aux", sample_name="no_sample", conn=source_extra_conn
)
```

```{code-cell} ipython3
meas = Measurement(exp=exp)
meas.register_parameter(my_inst.current)
meas.register_parameter(my_inst.voltage, setpoints=(my_inst.current,))

# Add 10 runs with gradually more and more data

for run_id in range(1, 11):
    with meas.run() as datasaver:
        for step, noise in enumerate(np.random.randn(run_id)):
            datasaver.add_result((my_inst.current, step), (my_inst.voltage, noise))
```

```{code-cell} ipython3
exp.data_set(3).guid
```

```{code-cell} ipython3
extract_runs_into_db(extra_source_path, target_path, 1, 3)
```

```{code-cell} ipython3
target_exp_aux = load_experiment_by_name(
    name="extract_runs_experiment_aux", conn=target_conn
)
```

The GUID should be preserved.

```{code-cell} ipython3
target_exp_aux.data_set(2).guid
```

And the original `run_id` is preserved as `captured_run_id`

```{code-cell} ipython3
target_exp_aux.data_set(2).captured_run_id
```

## Uniquely identifying and loading runs

As runs move from one database to the other, uniquely identifying a run becomes non-trivial.

+++

Note how we now have 2 runs in the same DB sharing the same `captured_run_id`. This means that `captured_run_id` is **not** a unique key. We can demonstrate that `captured_run_id` is not unique by looking up the `GUID`s that match this `captured_run_id`.

```{code-cell} ipython3
from qcodes.dataset import (
    load_by_guid,
    load_by_run_spec,
    get_guids_by_run_spec,
)
```

```{code-cell} ipython3
guids = get_guids_by_run_spec(conn=target_conn, captured_run_id=3)
guids
```

```{code-cell} ipython3
load_by_guid(guids[0], conn=target_conn)
```

```{code-cell} ipython3
load_by_guid(guids[1], conn=target_conn)
```

To enable loading of runs that may share the same `captured_run_id`, the function `load_by_run_data` is supplied.
This function takes one or more optional sets of metadata. If more than one run matching this information is found the metadata of the matching runs is printed and an error is raised. It is now possible to suply more information to the function to uniquely identify a specific run.

```{code-cell} ipython3
try:
    load_by_run_spec(captured_run_id=3, conn=target_conn)
except NameError:
    print("Caught a NameError")
```

To single out one of these two runs, we can thus specify the `experiment_name`:

```{code-cell} ipython3
load_by_run_spec(
    captured_run_id=3, experiment_name="extract_runs_experiment_aux", conn=target_conn
)
```
