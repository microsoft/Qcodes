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

# The Snapshot

This notebook sheds some light on the snapshot of instruments.

__NOTE__: this notebook uses a depreated `Loop` construct for some of its examples. Please, instead, refer to [__Working with snapshots__ notebook from `docs/examples/DataSet`](DataSet/Working%20with%20snapshots.ipynb).

```{code-cell} ipython3
from pprint import pprint
import json
import os

import qcodes as qc

from qcodes.tests.instrument_mocks import DummyInstrument
from qcodes.loops import Loop

# For this tutorial, we initialise our favourite pair of mock instruments,
# a DMM and a DAC

dmm = DummyInstrument('dmm', gates=['v1', 'v2'])
dac = DummyInstrument('dac', gates=['ch1', 'ch2'])

station = qc.Station(dmm, dac)
```

The main point of having a `Station` is that it *snapshots* the state of all added instruments. But what does that mean? Recall that an instrument is, loosely speaking, a collection of `Parameters`.

+++

## Parameter snapshot

```{code-cell} ipython3
# Each parameter has a snapshot, containing information about its current value,
# when that value was set, what the allowed values are, etc.
pprint(dmm.v1.snapshot())
```

## Instrument snapshot

```{code-cell} ipython3
# Each instrument has a snapshot that is basically the snapshots of all the parameters
pprint(dmm.snapshot())
```

## Sweep snapshot

```{code-cell} ipython3
# When running QCoDeS loops, something is being swept. This is controlled with the `sweep` of a parameter.
# Sweeps also have snapshots
a_sweep = dac.ch1.sweep(0, 10, num=25)
pprint(a_sweep.snapshot())
```

## Loop/Measurement snapshot

```{code-cell} ipython3
# All this is of course nice since a snapshot is saved every time a measurement is 
# performed. Let's see this in action with a Loop.

# This is a qc.loop, sweeping a dac gate and reading a dmm voltage
lp = Loop(dac.ch1.sweep(0, 1, num=10), 0).each(dmm.v1)

# before the loop runs, the snapshot is quite modest; it contains the snapshots of
# the two involved parameters and the sweep
pprint(lp.snapshot())
```

```{code-cell} ipython3
# After the loop has run, the dataset contains more information, in particular the 
# snapshots for ALL parameters off ALL instruments in the station
data = lp.run('data/dataset')
pprint(data.snapshot())

# This is the snapshot that get's saved to disk alongside your data. 
# It's worthwhile familiarising yourself with it, so that you may retrieve
# valuable information down the line!
```

```{code-cell} ipython3

```
