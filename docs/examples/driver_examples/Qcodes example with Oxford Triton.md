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

# QCoDeS Example with Oxford Triton

```{code-cell} ipython3
%matplotlib nbagg
import matplotlib.pyplot as plt
import time
import numpy as np
import qcodes as qc
```

```{code-cell} ipython3
import qcodes.instrument_drivers.oxford.triton as triton


triton = triton.Triton(name = 'Triton 1', address='127.0.0.1', port=33576, tmpfile='Triton1_thermometry.reg')
# triton._get_temp_channels('thermometry.reg')
# print(triton.chan_alias)

print(triton.time.get())
print(triton.T5.get())
print(triton.MC.get())
# for name,param in triton.parameters.items():
#     print(name,param.get())
print(triton.action.get())
print(triton.status.get())
# triton.close()
```

```{code-cell} ipython3
triton.get_idn()
```
