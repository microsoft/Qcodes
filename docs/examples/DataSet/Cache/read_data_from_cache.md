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

# Read data from cache

```{code-cell} ipython3
from qcodes import load_by_run_spec
import time
```

This notebook is meant to be used together with [Write data to cache](./write_for_caching.ipynb) to demonstate the use of the datasets cache interactively from another process. It demonstrates how data can be read gradually into the cache as the data is written by the other notebook. This is intended to be useful for plotting or other monitoring of the data as it is written. In the line below insert the ``captured_run_id`` of the dataset being captured in the other notebook.

The cache has the same format as the data read from `dataset.get_parameter_data`

```{code-cell} ipython3
dataset_run_id = 170
ds = load_by_run_spec(captured_run_id=dataset_run_id)
```

Here we will print the data as it is read from the database. As can be seen the number of datapoints grows as the data is written to the dataset and read back into the cache. Note that the cache only reads new datapoints and does not require the entire data to be reread from the database.

```{code-cell} ipython3
while True:
    time.sleep(0.5)
    print(ds.cache.data())
    if ds.completed:
        break
```
