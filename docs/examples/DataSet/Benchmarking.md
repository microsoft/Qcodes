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

# Dataset Benchmarking

This notebook is a behind-the-scenes benchmarking notebook, mainly for use by developers. The recommended way for users to interact with the dataset is via the `Measurement` object and its associated context manager. See the corresponding notebook for a comprehensive toturial on how to use those.

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt
import qcodes as qc
from qcodes import ParamSpec, new_data_set, new_experiment, initialise_database, load_or_create_experiment
import numpy as np
```

```{code-cell} ipython3
qc.config.core.db_location
```

```{code-cell} ipython3
initialise_database()
```

## Setup

```{code-cell} ipython3
exp = load_or_create_experiment("benchmarking", sample_name="the sample is a lie")
exp
```

Now we can create a dataset. Note two things:

    - if we don't specfiy a exp_id, but we have an experiment in the experiment container the dataset will go into that one.
    - dataset can be created from the experiment object


```{code-cell} ipython3
dataSet = new_data_set("benchmark_data", exp_id=exp.exp_id)
exp
```

In this benchmark we will assueme that we are doing a 2D loop and investigate the performance implications of writing to the dataset

```{code-cell} ipython3
x_shape = 100
y_shape = 100
```

## Baseline: Generate data

```{code-cell} ipython3
%%time
for x in range(x_shape):
    for y in range(y_shape):
        z = np.random.random_sample(1)
```

and store in memory

```{code-cell} ipython3
x_data = np.zeros((x_shape, y_shape))
y_data = np.zeros((x_shape, y_shape))
z_data = np.zeros((x_shape, y_shape))
```

```{code-cell} ipython3
%%time
for x in range(x_shape):
    for y in range(y_shape):
        x_data[x,y] = x
        y_data[x,y] = y
        z_data[x,y] = np.random.random_sample()
```

## Add to dataset inside double loop

```{code-cell} ipython3
double_dataset = new_data_set("doubledata", exp_id=exp.exp_id,
                              specs=[ParamSpec("x", "numeric"),
                                     ParamSpec("y", "numeric"),
                                     ParamSpec('z', "numeric")])
double_dataset.mark_started()
```

Note that this is so slow that we are only doing a 10th of the computation

```{code-cell} ipython3
%%time
for x in range(x_shape//10):
    for y in range(y_shape):
        double_dataset.add_results([{"x": x, 'y': y, 'z': np.random.random_sample()}])
```

## Add the data in outer loop and store as np array

```{code-cell} ipython3
single_dataset = new_data_set("singledata", exp_id=exp.exp_id,
                              specs=[ParamSpec("x", "array"),
                                     ParamSpec("y", "array"),
                                     ParamSpec('z', "array")])
single_dataset.mark_started()
x_data = np.zeros((y_shape))
y_data = np.zeros((y_shape))
z_data = np.zeros((y_shape))
```

```{code-cell} ipython3
%%time
for x in range(x_shape):
    for y in range(y_shape):
        x_data[y] = x
        y_data[y] = y
        z_data[y] = np.random.random_sample(1)
    single_dataset.add_results([{"x": x_data, 'y': y_data, 'z': z_data}])
```

## Save once after loop

```{code-cell} ipython3
zero_dataset = new_data_set("zerodata", exp_id=exp.exp_id,
                            specs=[ParamSpec("x", "array"),
                                   ParamSpec("y", "array"),
                                   ParamSpec('z', "array")])
zero_dataset.mark_started()
x_data = np.zeros((x_shape, y_shape))
y_data = np.zeros((x_shape, y_shape))
z_data = np.zeros((x_shape, y_shape))
```

```{code-cell} ipython3
%%time
for x in range(x_shape):
    for y in range(y_shape):
        x_data[x,y] = x
        y_data[x,y] = y
        z_data[x,y] = np.random.random_sample(1)
zero_dataset.add_results([{'x':x_data, 'y':y_data, 'z':z_data}])
```

## Array parameter

```{code-cell} ipython3
array1D_dataset = new_data_set("array1Ddata", exp_id=exp.exp_id,
                               specs=[ParamSpec("x", "array"),
                                      ParamSpec("y", "array"),
                                      ParamSpec('z', "array")])
array1D_dataset.mark_started()
y_setpoints = np.arange(y_shape)
```

```{code-cell} ipython3
%%timeit
for x in range(x_shape):
    x_data[x,:] = x
    array1D_dataset.add_results([{'x':x_data[x,:], 'y':y_setpoints, 'z':np.random.random_sample(y_shape)}])
```

```{code-cell} ipython3
x_data = np.zeros((x_shape, y_shape))
y_data = np.zeros((x_shape, y_shape))
z_data = np.zeros((x_shape, y_shape))
y_setpoints = np.arange(y_shape)
```

```{code-cell} ipython3
array0D_dataset = new_data_set("array0Ddata", exp_id=exp.exp_id,
                               specs=[ParamSpec("x", "array"),
                                      ParamSpec("y", "array"),
                                      ParamSpec('z', "array")])
array0D_dataset.mark_started()
```

```{code-cell} ipython3
%%timeit
for x in range(x_shape):
    x_data[x,:] = x
    y_data[x,:] = y_setpoints
    z_data[x,:] = np.random.random_sample(y_shape)
array0D_dataset.add_results([{'x':x_data, 'y':y_data, 'z':z_data}])
```

## Insert many

```{code-cell} ipython3
data = []
for i in range(100):
    for j in range(100):
        data.append({'x': i, 'y':j, 'z':np.random.random_sample()})
```

```{code-cell} ipython3
many_Data = new_data_set("many_data", exp_id=exp.exp_id,
                         specs=[ParamSpec("x", "numeric"),
                                ParamSpec("y", "numeric"),
                                ParamSpec("z", "numeric")])
many_Data.mark_started()
```

```{code-cell} ipython3
%%timeit
many_Data.add_results(data)
```
