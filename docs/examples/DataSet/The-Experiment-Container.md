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

# The Experiment Container

This notebook explains how the database works as an experiment container.

+++

## Required imports

```{code-cell} ipython3
import os
import qcodes as qc
from qcodes.dataset.sqlite.database import initialise_or_create_database_at, connect
from qcodes.dataset.data_set import new_data_set
from qcodes.dataset.experiment_container import (experiments, load_experiment,
                                                 load_last_experiment,new_experiment,
                                                 load_experiment_by_name,
                                                 load_or_create_experiment)
from qcodes.dataset.experiment_settings import (reset_default_experiment_id,
                                                get_default_experiment_id)
```

## The experiments inside the database

```{code-cell} ipython3
db_file_path = os.path.join(os.getcwd(), 'exp_container_tutorial.db')
initialise_or_create_database_at(db_file_path)
```

The database holds a certain number of **experiments**. They may be viewed:

```{code-cell} ipython3
experiments()
```

Not surprisingly, our new database is empty.

We now introduce a term that we call the **default experiment**. In short, it is the experiment that will be used for a QCoDeS `DataSet`, if the user do not explicitly pass an experiment into objects that create that `DataSet`. In another word, that `DataSet` will be belong to the default experiment. We do not want to go into the details of `DataSet` here, and refer to the [DataSet notebook](https://qcodes.github.io/Qcodes/examples/DataSet/DataSet-class-walkthrough.html) and [Performing measurements using qcodes parameters and dataset](https://qcodes.github.io/Qcodes/examples/DataSet/Performing-measurements-using-qcodes-parameters-and-dataset.html) for what we mean from `DataSet` and how we can pass an experiment explicitly.

By default, the last experiment in the database is the default experiment. The default experiment can be changed if another experiment in the database is created or loaded. We will explore this in this notebook. 

Users should not worry about checking the default experiment in their normal workflow, but in this notebook, we show how it works to let them have an idea what the default experiment is and how it changes.  

+++

We need a connection to our database to get which experiment is the default one:

```{code-cell} ipython3
conn = connect(db_file_path)
```

Because our database is empty now, asking for the default experiment will rasie an error asking to create an experiment. So, let's add some experiments to explore more:

```{code-cell} ipython3
exp_a = new_experiment('first_exp', sample_name='old_sample')
exp_b = new_experiment('second_exp', sample_name='slightly_newer_sample')
exp_c = load_or_create_experiment('third_exp', sample_name='brand_new_sample')
```

We recommend using the `load_or_create_experiment` function as the primary function dealing with experiments, not only because it is the most versatile function, but also because it can prevent creating duplicate experiments in one database.

```{code-cell} ipython3
experiments()
```

We notice that each experiment is labelled by an integer number, which is the `exp_id`. This ID can be used when looking up properties of each experiment.

+++

Let's check to see which experiment is the default now:

```{code-cell} ipython3
get_default_experiment_id(conn)
```

The latest created or loaded experiment in the database becomes the default experiment, and the function returns the `exp_id` of that experiment, which in this case it is `exp_c` with `exp_id` of 3.

+++

Let us add some `DataSet` to our experiments. For the sake of clarity, we don't add any data to the `DataSet` here, and refer to the above-mentioned notebooks for the details. Note that the `new_data_set` function is used here ONLY for the sake of exercise and should NOT be used in the actual experiment.

```{code-cell} ipython3
new_data_set('run_a')
```

Since the default experiment is exp_c (`exp_id`=3), the above `DataSet` belongs to this experiment.

```{code-cell} ipython3
exp_c
```

Let's load another experiment (`exp_b`). We know that the latest created/ loaded experiment should be the default experiment, meaning any new `DataSet` should belong to this experiment:

```{code-cell} ipython3
load_or_create_experiment('second_exp', sample_name='slightly_newer_sample')
```

Let's confirm that actually the second experiment (`exp_id`=2) is the default now:

```{code-cell} ipython3
get_default_experiment_id(conn)
```

```{code-cell} ipython3
new_data_set('first_run_b')
```

```{code-cell} ipython3
new_data_set('second_run_b')
```

Two above `DataSet`s should belong to `exp_b`:

```{code-cell} ipython3
exp_b
```

We can also explicitly use `exp_id` in creating `DataSet`s, so let's add a `DataSet` to the first experiment:

```{code-cell} ipython3
new_data_set('first_run', exp_id=1)
```

```{code-cell} ipython3
exp_a
```

The default experiment gets reset upon initialization of a database. Let's check this by initializing our database again (note that our database is not empty anymore):

```{code-cell} ipython3
initialise_or_create_database_at(db_file_path)
```

The default experiment was `exp_id`=2. As we initialized our database again, the default experiment has been reset, meaning the last experiment in the database should be the default one now (we know the last experiment in the database is `exp_id`=3). Let's check this:

```{code-cell} ipython3
get_default_experiment_id(conn)
```

Users may not need to use the reset function explicitly, but in the case they want to use it, here we show how to do that:

+++

First, we load an experiment other than the last experiment and check the default experiment is the just loaded experiment:

```{code-cell} ipython3
load_experiment(1)
```

```{code-cell} ipython3
get_default_experiment_id(conn)
```

Now, we reset the default experiment and expect to see the last experiment (`exp_id`=3) is the default one:

```{code-cell} ipython3
reset_default_experiment_id() # the explicit database connection can be used as an optional argument 
```

```{code-cell} ipython3
get_default_experiment_id(conn)
```

Let's make sure it is truly the default experiment by creating a new `DataSet`:

```{code-cell} ipython3
new_data_set('default_run')
```

This `DataSet` should belong to `exp_c`:

```{code-cell} ipython3
exp_c
```

There are a few other useful functions to load experiments:

```{code-cell} ipython3
load_experiment_by_name('second_exp', sample='slightly_newer_sample')  # loads using name and sample
```

```{code-cell} ipython3
load_last_experiment() # loads the last experiment in the database
```
