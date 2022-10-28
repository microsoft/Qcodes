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

# Pedestrian example of subscribing to a DataSet

It is possible to *subscribe* to a dataset. Subscribing means adding a function to the dataset and having the dataset call that function every time a result is added to the dataset (or more rarely, see below).

### Call signature

The subscribing function must have the following call signature:
```
fun(results: List[Tuple[Value]], length: int,
    state: Union[MutableSequence, MutableMapping]) -> None:
    """
    Args:
        results: A list of tuples where each tuple holds the results inserted into the dataset.
            For two scalar parameters, X and Y, results might look like [(x1, y1), (x2, y2), ...]
        length: The current length of the dataset.
        state: Any mutable sequence/mapping that can be used to hold information from call to call.
            In practice a list or a dict.
    """
```
Below we provide an example function that counts the number of times a voltage has exceeded a certain limit.

### Frequency

Since calling the function **every** time an insertion is made may be too frequent, a `min_wait` and a `min_count` argument may be provided when subscribing. The dataset will then only call the function upon inserting a result
if `min_wait` seconds have elapsed since the last call (or the start of the subscription, in the time before the first call) and `min_count` results have been added to the dataset since the last call (or the start of the subscription). All the results added in the meantime are queued and passed to the function in one go.

### Order

The subscription must be set up **after** all parameters have been added to the dataset.

```{code-cell} ipython3
from qcodes import initialise_database, load_or_create_experiment, \
    new_data_set, ParamSpec
import numpy as np
from time import sleep
```

## Example 1: A notification

We imagine scanning a frequency and reading out a noisy voltage. When the voltage has exceeded a threshold 5 times, we want to receive a warning. Let us initialise our database and create an experiment.

```{code-cell} ipython3
initialise_database()
exp = load_or_create_experiment(experiment_name="subscription_tutorial", sample_name="no_sample")
```

```{code-cell} ipython3
dataSet = new_data_set("test",
                       exp_id=exp.exp_id,
                       specs=[ParamSpec("x", "numeric", unit='Hz'),
                              ParamSpec("y", "numeric", unit='V')])
dataSet.mark_started()
```

```{code-cell} ipython3
def threshold_notifier(results, length, state):
    if len(state) > 4:
        print(f'At step {length}: The voltage exceeded the limit 5 times! ')
        state.clear()
    for result in results:
        if result[1] > 0.8:
            state.append(result[1])
```

```{code-cell} ipython3
# now perform the subscription
# since this is important safety info, we want our callback function called
# on EVERY insertion
sub_id = dataSet.subscribe(threshold_notifier, min_wait=0, min_count=1, state=[])
```

```{code-cell} ipython3
for x in np.linspace(100, 200, 150):
    y = np.random.randn()
    dataSet.add_results([{"x": x, "y": y}])
```

```{code-cell} ipython3
dataSet.unsubscribe_all()
```

## Example 2: ASCII Plotter

While this example does not represent a data acqusition that one may deal with in a real experiment, it is a fun practice of subscribing to a dataset.

```{code-cell} ipython3
dataSet = new_data_set("test", exp_id=exp.exp_id,
                       specs=[ParamSpec("blip", "numeric", unit='bit'),
                              ParamSpec("blop", "numeric", unit='bit')])
dataSet.mark_started()
```

```{code-cell} ipython3
def ASCII_plotter_5bit(results, length, state):
    """
    Glorious 5-bit signal plotter
    
    Digitises the range (-1, 1) with 4 bits and plots it
    in stdout. Crashes and burns if given data outside that
    interval.
    """
    for result in results:
        plotline = ['.'] * 32
        yvalue = result[1]
        yvalue += 1
        yvalue /= 2
        yvalue = int(yvalue*31)
        plotline[yvalue] = 'O'
        print(''.join(plotline))
        
```

```{code-cell} ipython3
sub_id = dataSet.subscribe(ASCII_plotter_5bit, min_wait=0, min_count=3, state=[])
```

```{code-cell} ipython3
for x in np.linspace(0, 3*np.pi, 100):
    yvalue = 0.9*np.sin(x) + np.random.randn()*0.05
    dataSet.add_results([{"blip": x, "blop": yvalue}])
    sleep(0.1)
```
