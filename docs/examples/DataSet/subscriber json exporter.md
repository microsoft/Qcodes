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

# Subscriber with JSON export

__NOTE__: this is an __outdated__ notebook, some of the functions that are used here are considered __private__ to QCoDeS and are not intended for use by users (for example, `DataSet.subscribe`). This notebook will be re-written in the future.

```{code-cell} ipython3
import logging
import copy
import numpy as np
import json
```

```{code-cell} ipython3
from qcodes import load_or_create_experiment, new_data_set, ParamSpec
from qcodes.dataset.json_exporter import \
    json_template_heatmap, json_template_linear, \
    export_data_as_json_heatmap, export_data_as_json_linear
```

```{code-cell} ipython3
logging.basicConfig(level="INFO")
```

```{code-cell} ipython3
exp = load_or_create_experiment('json-export-subscriber-test', 'no-sample')
```

```{code-cell} ipython3
dataSet = new_data_set("test",
                       exp_id=exp.exp_id,
                       specs=[ParamSpec("x", "numeric"), ParamSpec("y", "numeric")])
dataSet.mark_started()
```

```{code-cell} ipython3
mystate = {}
mystate['json'] = copy.deepcopy(json_template_linear)
mystate['json']['x']['name'] = 'xname'
mystate['json']['x']['unit'] = 'xunit'
mystate['json']['x']['full_name'] = 'xfullname'
mystate['json']['y']['name'] = 'yname'
mystate['json']['y']['unit'] = 'yunit'
mystate['json']['y']['full_name'] = 'yfullname'
```

```{code-cell} ipython3
sub_id = dataSet.subscribe(export_data_as_json_linear, min_wait=0, min_count=20,
                           state=mystate, callback_kwargs={'location': 'foo'})
```

```{code-cell} ipython3
s = dataSet.subscribers[sub_id]
```

```{code-cell} ipython3
mystate
```

```{code-cell} ipython3
for x in range(100):
    y = x
    dataSet.add_results([{"x":x, "y":y}])
dataSet.mark_completed()
```

```{code-cell} ipython3
mystate
```

```{code-cell} ipython3
mystate = {}
xlen = 5
ylen = 10
mystate['json'] = json_template_heatmap.copy()
mystate['data'] = {}
mystate['data']['xlen'] = xlen
mystate['data']['ylen'] = ylen
mystate['data']['x'] = np.zeros((xlen*ylen), dtype=np.object)
mystate['data']['x'][:] = None
mystate['data']['y'] = np.zeros((xlen*ylen), dtype=np.object)
mystate['data']['y'][:] = None
mystate['data']['z'] = np.zeros((xlen*ylen), dtype=np.object)
mystate['data']['z'][:] = None
mystate['data']['location'] = 0
```

```{code-cell} ipython3
dataSet_hm = new_data_set("test", exp_id=exp.exp_id,
                          specs=[ParamSpec("x", "numeric"),
                                 ParamSpec("y", "numeric"),
                                 ParamSpec("z", "numeric")])
dataSet_hm.mark_started()
```

```{code-cell} ipython3
sub_id = dataSet_hm.subscribe(export_data_as_json_heatmap, min_wait=0, min_count=20,
                              state=mystate, callback_kwargs={'location': './foo'})
```

```{code-cell} ipython3
for x in range(xlen):
    for y in range(ylen):
        z = x+y
        dataSet_hm.add_results([{"x":x, "y":y, 'z':z}])
dataSet_hm.mark_completed()
```

```{code-cell} ipython3
mystate['json']
```
