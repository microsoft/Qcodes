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

# Databases

This is an example notebook to briefly discuss the use of SQLite3 databases in QCoDeS. Generally, databases are used to store values of `parameters` obtained during a `measurement` in `DataSet` ojects. Specifically, storage occurs as part of  the `add_result` method of the `datasaver` object in the context manager. `DataSets` can also be read from the database for analysis.  

## Imports

For our example we require only a few modules.


```{code-cell} ipython3
import qcodes as qc
from qcodes import (
    initialise_database,
    initialise_or_create_database_at,
)
```

## Initialization, creation and upgrade

A database can be initialized with a simple function call, `initialise_database`, which will initialize a database at the default location which is normally `~/experiments.db`. This function produces does not return or print a value, but the location of the currently initialized database is stored in the QCoDeS configuration `qc.config.core.db_location`.


```{code-cell} ipython3
initialise_database()

print(qc.config.core.db_location)
```

When an existing database is initialized, it is always examined and upgraded to the latest version by a tested automatic process. When this occurs, a more detailed output is produced that details the upgrade process. This upgrade occurs with journaling to ensure data safety, to date there have been no instances of data corruption during database upgrades. 

Example database upgrade output:
> Upgrading database; v0 -> v1: : 0it [00:00, ?it/s]

> Upgrading database; v1 -> v2: 100%|██████████| 1/1 [00:00<00:00, 561.49it/s]

> Upgrading database; v2 -> v3: : 0it [00:00, ?it/s]

> Upgrading database; v3 -> v4: : 0it [00:00, ?it/s]

> Upgrading database; v4 -> v5: 100%|██████████| 1/1 [00:00<00:00, 925.08it/s]

> Upgrading database; v5 -> v6: : 0it [00:00, ?it/s]

> Upgrading database; v6 -> v7: 100%|██████████| 1/1 [00:00<00:00, 360.49it/s]

> Upgrading database; v7 -> v8: 100%|██████████| 1/1 [00:00<00:00, 850.94it/s]

> Upgrading database; v8 -> v9: 100%|██████████| 1/1 [00:00<00:00, 1059.97it/s]

+++

Frequently, new databases will need to be created to meet the needs of data storage and organization. Moreover, these should be created a particular locations (e.g. a specific drive) with particular names (e.g. QCoDeS_example.db). These operations are all executed in an intelligent way by the `initialise_or_create_database_at` function. 

> Note that this will only change the location of the database in the current session not the default location stored in the qcodes config file. This means that this function 
is well suited to be part of a measurement script that ensures all data measured goes into the correct db but not to set the default location.

> Note: This function requires a path to exist, i.e. it cannot create folders.

```{code-cell} ipython3
initialise_or_create_database_at("./QCoDeS_example.db")

print(qc.config.core.db_location)
```

When databases are always created as a `v1` database and then upgraded to the current version as detailed in the output. If this function is called again to initalize the database, then no messages are printed.

```{code-cell} ipython3
initialise_or_create_database_at("./QCoDeS_example.db")

print(qc.config.core.db_location)
```

> Note: Later calls for initializing a database via the `initalise_database` function will simply initialize the database listed at qc.config.core.db_location.

+++


> Note: The database path stored in `qc.config.core.db_location` may be changed to change the location of the default database.

+++

Some data management options--including the ability to save data in memory only--included as options of the  `DataSet` class:
- [Saving data in-memory only example](InMemoryDataSet.ipynb)
- [Saving data by a background thread](Saving_data_in_the_background.ipynb)

Moreover, we have also written an [example notebook](Extracting-runs-from-one-DB-file-to-another.ipynb) of transferring `DataSets` between database flies that may serve as a template for more complex data organization protocols.
