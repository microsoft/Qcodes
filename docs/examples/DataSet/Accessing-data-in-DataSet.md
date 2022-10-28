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

# Accessing data in a DataSet

After a measurement is completed all the acquired data and metadata around it is accessible via a `DataSet` object. This notebook presents the useful methods and properties of the `DataSet` object which enable convenient access to the data, parameters information, and more. For general overview of the `DataSet` class, refer to [DataSet class walkthrough](DataSet-class-walkthrough.ipynb).

+++

## Preparation: a DataSet from a dummy Measurement

In order to obtain a `DataSet` object, we are going to run a `Measurement` storing some dummy data (see notebook on [Performing measurements using qcodes parameters and dataset](Performing-measurements-using-qcodes-parameters-and-dataset.ipynb) notebook for more details).

```{code-cell} ipython3
import tempfile
import os

import numpy as np

import qcodes
from qcodes import initialise_or_create_database_at, \
    load_or_create_experiment, Measurement, Parameter, \
    Station
from qcodes.dataset.plotting import plot_dataset
```

```{code-cell} ipython3
db_path = os.path.join(tempfile.gettempdir(), 'data_access_example.db')
initialise_or_create_database_at(db_path)

exp = load_or_create_experiment(experiment_name='greco', sample_name='draco')
```

```{code-cell} ipython3
x = Parameter(name='x', label='Voltage', unit='V',
              set_cmd=None, get_cmd=None)
t = Parameter(name='t', label='Time', unit='s',
              set_cmd=None, get_cmd=None)
y = Parameter(name='y', label='Voltage', unit='V',
              set_cmd=None, get_cmd=None)
y2 = Parameter(name='y2', label='Current', unit='A',
               set_cmd=None, get_cmd=None)
q = Parameter(name='q', label='Qredibility', unit='$',
               set_cmd=None, get_cmd=None)
```

```{code-cell} ipython3
meas = Measurement(exp=exp, name='fresco')

meas.register_parameter(x)
meas.register_parameter(t)
meas.register_parameter(y, setpoints=(x, t))
meas.register_parameter(y2, setpoints=(x, t))
meas.register_parameter(q)  # a standalone parameter

x_vals = np.linspace(-4, 5, 50)
t_vals = np.linspace(-500, 1500, 25)

with meas.run() as datasaver:
    for xv in x_vals:
        for tv in t_vals:
            yv = np.sin(2*np.pi*xv)*np.cos(2*np.pi*0.001*tv) + 0.001*tv
            y2v = np.sin(2*np.pi*xv)*np.cos(2*np.pi*0.001*tv + 0.5*np.pi) - 0.001*tv
            datasaver.add_result((x, xv), (t, tv), (y, yv), (y2, y2v))
        q_val = np.max(yv) - np.min(y2v)  # a meaningless value
        datasaver.add_result((q, q_val))

dataset = datasaver.dataset
```

For the sake of demonstrating what kind of data we've produced, let's use `plot_dataset` to make some default plots of the data.

```{code-cell} ipython3
plot_dataset(dataset)
```

## DataSet indentification

Before we dive into what's in the `DataSet`, let's briefly note how a `DataSet` is identified.

```{code-cell} ipython3
dataset.captured_run_id
```

```{code-cell} ipython3
dataset.exp_name
```

```{code-cell} ipython3
dataset.sample_name
```

```{code-cell} ipython3
dataset.name
```

## Parameters in the DataSet

In this section we are getting information about the parameters stored in the given `DataSet`.

> Why is that important? Let's jump into *data*!

As it turns out, just "arrays of numbers" are not enough to reason about a given `DataSet`. Even comping up with a reasonable deafult plot, which is what `plot_dataset` does, requires information on `DataSet`'s parameters. In this notebook, we first have a detailed look at what is stored about parameters and how to work with this information. After that, we will cover data access methods.

+++

### Run description

Every dataset comes with a "description" (aka "run description"):

```{code-cell} ipython3
dataset.description
```

The description, an instance of `RunDescriber` object, is intended to describe the details of a dataset. In the future releases of QCoDeS it will likely be expanded. At the moment, it only contains an `InterDependencies_` object under its `interdeps` attribute - which stores all the information about the parameters of the `DataSet`.

Let's look into this `InterDependencies_` object.

+++

### Interdependencies

`Interdependencies_` object inside the run description contains information about all the parameters that are stored in the `DataSet`. Subsections below explain how the individual information about the parameters as well as their relationships are captured in the `Interdependencies_` object.

```{code-cell} ipython3
interdeps = dataset.description.interdeps
interdeps
```

#### Dependencies, inferences, standalones

+++

Information about every parameter is stored in the form of `ParamSpecBase` objects, and the releationship between parameters is captured via `dependencies`, `inferences`, and `standalones` attributes.

For example, the dataset that we are inspecting contains no inferences, and one standalone parameter `q`, and two dependent parameters `y` and `y2`, which both depend on independent `x` and `t` parameters:

```{code-cell} ipython3
interdeps.inferences
```

```{code-cell} ipython3
interdeps.standalones
```

```{code-cell} ipython3
interdeps.dependencies
```

`dependencies` is a dictionary of `ParamSpecBase` objects. The keys are dependent parameters (those which depend on other parameters), and the corresponding values in the dictionary are tuples of independent parameters that the dependent parameter in the key depends on. Coloquially, each key-value pair of the `dependencies` dictionary is sometimes referred to as "parameter tree".

`inferences` follows the same structure as `dependencies`.

`standalones` is a set - an unordered collection of `ParamSpecBase` objects representing "standalone" parameters, the ones which do not depend on other parameters, and no other parameter depends on them.

+++

#### ParamSpecBase objects

+++

`ParamSpecBase` object contains all the necessary information about a given parameter, for example, its `name` and `unit`:

```{code-cell} ipython3
ps = list(interdeps.dependencies.keys())[0]
print(f'Parameter {ps.name!r} is in {ps.unit!r}')
```

`paramspecs` property returns a tuple of `ParamSpecBase`s for all the parameters contained in the `Interdependencies_` object:

```{code-cell} ipython3
interdeps.paramspecs
```

Here's a trivial example of iterating through dependent parameters of the `Interdependencies_` object and extracting information about them from the `ParamSpecBase` objects:

```{code-cell} ipython3
for d in interdeps.dependencies.keys():
    print(f'Parameter {d.name!r} ({d.label}, {d.unit}) depends on:')
    for i in interdeps.dependencies[d]:
        print(f'- {i.name!r} ({i.label}, {i.unit})')
```

#### Other useful methods and properties

+++

`Interdependencies_` object has a few useful properties and methods which make it easy to work it and with other `Interdependencies_` and `ParamSpecBase` objects.

For example, `non_dependencies` returns a tuple of all dependent parameters together with standalone parameters:

```{code-cell} ipython3
interdeps.non_dependencies
```

`what_depends_on` method allows to find what parameters depend on a given parameter:

```{code-cell} ipython3
t_ps = interdeps.paramspecs[2]
t_deps = interdeps.what_depends_on(t_ps)

print(f'Following parameters depend on {t_ps.name!r} ({t_ps.label}, {t_ps.unit}):')
for t_dep in t_deps:
    print(f'- {t_dep.name!r} ({t_dep.label}, {t_dep.unit})')
```

### Shortcuts to important parameters

For the frequently needed groups of parameters, `DataSet` object itself provides convenient methods and properties.

For example, use `dependent_parameters` property to get only dependent parameters of a given `DataSet`:

```{code-cell} ipython3
dataset.dependent_parameters
```

This is equivalent to:

```{code-cell} ipython3
tuple(dataset.description.interdeps.dependencies.keys())
```

### Note on inferences

Inferences between parameters is a feature that has not been used yet within QCoDeS. The initial concepts around `DataSet` included it in order to link parameters that are not directly dependent on each other as "dependencies" are. It is very likely that "inferences" will be eventually deprecated and removed.

+++

### Note on ParamSpec's

> `ParamSpec`s originate from QCoDeS versions prior to `0.2.0` and for now are kept for backwards compatibility. `ParamSpec`s are completely superseded by `InterDependencies_`/`ParamSpecBase` bundle and will likely be deprecated in future versions of QCoDeS together with the `DataSet` methods/properties that return `ParamSpec`s objects.

In addition to the `Interdependencies_` object, `DataSet` also holds `ParamSpec` objects (not to be confused with `ParamSpecBase` objects from above). Similar to `Interdependencies_` object, the `ParamSpec` objects hold information about parameters and their interdependencies but in a different way: for a given parameter, `ParamSpec` object itself contains information on names of parameters that it depends on, while for the `InterDependencies_`/`ParamSpecBase`s this information is stored only in the `InterDependencies_` object.

+++

`DataSet` exposes `paramspecs` property and `get_parameters()` method, both of which return `ParamSpec` objects of all the parameters of the dataset, and are not recommended for use:

```{code-cell} ipython3
dataset.paramspecs
```

```{code-cell} ipython3
dataset.get_parameters()
```

```{code-cell} ipython3
dataset.parameters
```

To give an example of what it takes to work with `ParamSpec` objects as opposed to `Interdependencies_` object, here's a function that one needs to write in order to find standalone `ParamSpec`s from a given list of `ParamSpec`s:

```{code-cell} ipython3
def get_standalone_parameters(paramspecs):
    all_independents = set(spec.name
                           for spec in paramspecs
                           if len(spec.depends_on_) == 0)
    used_independents = set(d for spec in paramspecs for d in spec.depends_on_)
    standalones = all_independents.difference(used_independents)
    return tuple(ps for ps in paramspecs if ps.name in standalones)

all_parameters = dataset.get_parameters()
standalone_parameters = get_standalone_parameters(all_parameters)
standalone_parameters
```

## Getting data from DataSet

In this section methods for retrieving the actual data from the `DataSet` are discussed.

### `get_parameter_data` - the powerhorse

`DataSet` provides one main method of accessing data - `get_parameter_data`. It returns data for groups of dependent-parameter-and-its-independent-parameters in a form of a nested dictionary of `numpy` arrays:

```{code-cell} ipython3
dataset.get_parameter_data()
```

#### Avoid excessive calls to loading data

Note that this call actually reads the data of the `DataSet` and in case of a `DataSet` with a lot of data can take noticable amount of time. Hence, it is recommended to limit the number of times the same data gets loaded in order to speed up the user's code.

+++

#### Loading data of selected parameters

Sometimes data only for a particular parameter or parameters needs to be loaded. For example, let's assume that after inspecting the `InterDependencies_` object from `dataset.description.interdeps`, we concluded that we want to load data of the `q` parameter and the `y2` parameter. In order to do that, we just pass the names of these parameters, or their `ParamSpecBase`s to `get_parameter_data` call:

```{code-cell} ipython3
q_param_spec = list(interdeps.standalones)[0]
q_param_spec
```

```{code-cell} ipython3
y2_param_spec = interdeps.non_dependencies[-1]
y2_param_spec
```

```{code-cell} ipython3
dataset.get_parameter_data(q_param_spec, y2_param_spec)
```

### `to_pandas_dataframe_dict` and `to_pandas_dataframe` - for `pandas` fans

`DataSet` provides two methods for accessing data with `pandas` - `to_pandas_dataframe` and  `to_pandas_dataframe_dict`. The method `to_pandas_dataframe_dict` returns data for groups of dependent-parameter-and-its-independent-parameters in a form of a dictionary of `pandas.DataFrame` s, while `to_pandas_dataframe` returns a concatendated `pandas.DataFrame` for groups of dependent-parameter-and-its-independent-parameters:

```{code-cell} ipython3
df_dict = dataset.to_pandas_dataframe_dict()

# For the sake of making this article more readable,
# we will print the contents of the `dfs` dictionary
# manually by calling `.head()` on each of the DataFrames
for parameter_name, df in df_dict.items():
    print(f"DataFrame for parameter {parameter_name}")
    print("-----------------------------")
    print(f"{df.head()!r}")
    print("")
```

Alternativly to concatinate the DataSet data into a single pandas Dataframe run the following:

```{code-cell} ipython3
df = dataset.to_pandas_dataframe()
print(f"{df.head()!r}")
```

Similar to `get_parameter_data`, `to_pandas_dataframe_dict` and `to_pandas_dataframe_dict` also supports retrieving data for a given parameter(s), as well as `start`/`stop` arguments.

Both `to_pandas_dataframe` and `to_pandas_dataframe_dict` is implemented based on `get_parameter_data`, hence the performance considerations mentioned above for `get_parameter_data` apply to these methods as well.

For more details on `to_pandas_dataframe` refer to [Working with pandas and xarray article](Working-With-Pandas-and-XArray.ipynb).

+++

### Exporting to other file formats

The dataset support exporting to netcdf and csv via the `dataset.export` method. See [Exporting QCoDes Datasets](./Exporting-data-to-other-file-formats.ipynb) for more information.

+++

### Data extraction into "other" formats

If the user desires to export a QCoDeS `DataSet` into a format that is not readily supported by `DataSet` methods, we recommend to use `to_pandas_dataframe_dict` or `to_pandas_dataframe_dict` first, and then convert the resulting `DataFrame` s into a the desired format. This is becuase `pandas` package already implements converting `DataFrame` to various popular formats including comma-separated text file (`.csv`), HDF (`.hdf5`), xarray, Excel (`.xls`, `.xlsx`), and more; refer to [Working with pandas and xarray article](Working-With-Pandas-and-XArray.ipynb), and [`pandas` documentation](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#serialization-io-conversion) for more information.

Refer to the docstrings of those methods for more information on how to use them.

```{code-cell} ipython3

```
