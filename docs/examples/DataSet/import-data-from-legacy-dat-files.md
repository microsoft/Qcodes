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

# Importing data from legacy .dat files

+++

This notebook shows you how you can import data from a legacy dataset generated with the old `qcodes.data` dataset. These are typically generated with `qcodes.loops.Loop` or `qcodes.measure.Measure`.

```{code-cell} ipython3
%matplotlib inline
import qcodes as qc
from qcodes.dataset.legacy_import import import_dat_file

from qcodes import initialise_database
from qcodes import load_or_create_experiment
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.plotting import plot_by_id
from qcodes.dataset.data_set import load_by_id

import numpy as np
```

```{code-cell} ipython3
# in case it was not there already
initialise_database()
# put the old data in a new experiment
exp = load_or_create_experiment('old_data_loading', sample_name='no_sample')
```

```{code-cell} ipython3
location2d = '../../../qcodes/tests/dataset/fixtures/data_2018_01_17/data_002_2D_test_15_43_14'
location1d = '../../../qcodes/tests/dataset/fixtures/data_2018_01_17/data_001_testsweep_15_42_57'
```

```{code-cell} ipython3
run_ids = import_dat_file(location1d, exp=exp)
axs, cbaxs = plot_by_id(run_ids[0])
```

```{code-cell} ipython3
run_ids = import_dat_file(location2d, exp=exp)
axs, cbaxs = plot_by_id(run_ids[0])
```

```{code-cell} ipython3

```
