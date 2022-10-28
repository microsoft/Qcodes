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

# Datasaving Examples

```{code-cell} ipython3
%matplotlib nbagg
import numpy as np
from importlib import reload
import qcodes as qc

from qcodes.loops import Loop
```

```{code-cell} ipython3
import sys
import logging
# Create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create STDERR handler
handler = logging.StreamHandler(sys.stderr)
# ch.setLevel(logging.DEBUG)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Set STDERR handler as the only handler 
logger.handlers = [handler]
```

```{code-cell} ipython3
from qcodes.data import hdf5_format
reload(hdf5_format)
h5fmt = hdf5_format.HDF5Format()
```

## Start a loop and generate data from dummy instruments 

```{code-cell} ipython3
station = qc.station.Station()
```

```{code-cell} ipython3
from qcodes.tests.instrument_mocks import MockParabola
station.add_component(MockParabola(name='MockParabola'))
```

```{code-cell} ipython3
loop = Loop(station.MockParabola.x[-100:100:20]).each(station.MockParabola.skewed_parabola)
data_l = loop.run(name='MockParabola_run', formatter=qc.data.gnuplot_format.GNUPlotFormat())

```

```{code-cell} ipython3
reload(hdf5_format)
h5fmt = hdf5_format.HDF5Format()
loop = Loop(station.MockParabola.x[-100:100:20]).loop(
    station.MockParabola.y[-100:50:10]).each(station.MockParabola.skewed_parabola)
data_l = loop.run(name='MockParabola_run', formatter=h5fmt)
```

```{code-cell} ipython3
from importlib import reload
from qcodes.data import hdf5_format
reload(hdf5_format)
h5fmt = hdf5_format.HDF5Format()
data2 = qc.data.data_set.DataSet(location=data_l.location, formatter=h5fmt)
data2.read()
```
