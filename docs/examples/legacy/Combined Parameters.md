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

# Combined Parameters

```{code-cell} ipython3
import qcodes as qc
import numpy as np 

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils.validators import Numbers
from qcodes.loops import Loop
```

If you want to sweep multiple parameters at once qcodes offers the combine function.
You can combine any number of any kind paramter. 
We'll use a ManualParameter for this example.

```{code-cell} ipython3
p1 = ManualParameter('p1', vals=Numbers(-10, 10))
p2 = ManualParameter('p2', vals=Numbers(-10, 10))
p3 = ManualParameter('p3', vals=Numbers(-10, 10))
p4 = ManualParameter('p4', vals=Numbers(-10, 10))
# set to -1 so we get some data out
p4.set(-1)
```

##  Simple combined  parameters 

```{code-cell} ipython3
combined = qc.combine(p1, p2, p3, name='combined')

sweep_vals = np.array([[1, 1,1], [1, 1,1]])
```

```{code-cell} ipython3
# 2d loop with a inner loop over a combined parameter
loop = Loop(p1.sweep(0,10,1)).loop(combined.sweep(sweep_vals), delay=0.001).each(p4)
```

```{code-cell} ipython3
data = loop.get_data_set(name='testsweep')
```

```{code-cell} ipython3
data = loop.run()
```

The combined_set just stores the indices 

```{code-cell} ipython3
print(data.combined_set)
```

But the acutal set values are saved, but labeled as "measured"

```{code-cell} ipython3
data.p3
```

## Combine and aggregate parameters

If an aggregator function is given, the aggregated values are saved instead of the indices.

```{code-cell} ipython3
# define an aggregator function that takes as arguments the parameters you whish to aggegate
def linear(x,y,z):
    return x+y+z
```

```{code-cell} ipython3
combined = qc.combine(p1, p2, p3, name='combined', label="Sum", unit="a.u", aggregator=linear)

x_vals = np.linspace(1, 2, 2)
y_vals = np.linspace(1, 2, 2)
z_vals = np.linspace(1, 2, 2)

```

```{code-cell} ipython3
# 2d loop with a inner loop over a combined parameter
loop = Loop(p1.sweep(0,10,1)).loop(combined.sweep(x_vals, y_vals, z_vals), delay=0.001).each(p4)
```

```{code-cell} ipython3
data = loop.get_data_set(name='testsweep')
```

```{code-cell} ipython3
data = loop.run()
```

the combined_set now stores the aggregated values

```{code-cell} ipython3
print(data.combined_set)
```

```{code-cell} ipython3
# snapshot of the combined parameter
combined.snapshot()
```
