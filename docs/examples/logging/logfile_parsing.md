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

# Logfile parsing
(back to overview [offline](../Main.ipynb),[online](https://nbviewer.jupyter.org/github/QCoDeS/Qcodes/tree/master/docs/examples/Main.ipynb))

[read on nbviewer](https://nbviewer.jupyter.org/github/QCoDeS/Qcodes/tree/master/docs/examples/logging/logfile_parsing.ipynb)

+++

Here, we provide an example concerning how to benefit from QCoDeS logs by simple analysis and visualisation.

```{code-cell} ipython3
%matplotlib inline
import dateutil
import os

import pandas as pd
import matplotlib.pyplot as plt

import qcodes as qc
from qcodes.logger import logfile_to_dataframe, time_difference, log_to_dataframe
from zipfile import ZipFile
```

```{code-cell} ipython3
# put the 30MB into a zip file
filepath = os.path.join(os.getcwd(), 'static', 'pythonlog.zip')
with ZipFile(filepath) as z:
    with z.open('pythonlog.log', 'r') as f:
        my_log = [line.decode() for line in f]
```

```{code-cell} ipython3
os.path.exists(filepath)
```

```{code-cell} ipython3
logdata = log_to_dataframe(my_log, separator=' - ', columns=['time', 'module', 'function', 'loglevel', 'message'])
```

The `logdata` is, now, a nice and tidy `DataFrame` that one can further manipulate to extract more information, if needed.

```{code-cell} ipython3
logdata
```

```{code-cell} ipython3
data = logdata
```

### Get the query time for the SR860

We know that the log file documents an experiment quering an SR860 for R and amplitude over and over. Let us analyse and visualise query response times.

```{code-cell} ipython3
qstr_R = 'Querying instrument SR860_120: OUTP\? 2'  # remember to escape
queries_R = data[data.message.str.contains(qstr_R)]
responses_R = data.loc[queries_R.index + 1]

qstr_lvl = 'Querying instrument SR860_120: SLVL\?'  # remember to escape
queries_lvl = data[data.message.str.contains(qstr_lvl)][:-1]
responses_lvl = data.loc[queries_lvl.index + 1]
```

### Find the elapsed times

```{code-cell} ipython3
elapsed_times_R = time_difference(queries_R.time, responses_R.time)
elapsed_times_lvl =  time_difference(queries_lvl.time, responses_lvl.time)
```

## Visualise!

```{code-cell} ipython3
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

fig, ax = plt.subplots(1,1)
ax.plot(queries_R.time.str.replace(',', '.').astype("datetime64[ns]"), elapsed_times_R, '.', label='R')
ax.plot(queries_lvl.time.str.replace(',', '.').astype("datetime64[ns]"), elapsed_times_lvl, '.', label='LVL')
fig.autofmt_xdate()
ax.set_ylabel('Response time (s)')
plt.legend()
```

```{code-cell} ipython3

```
