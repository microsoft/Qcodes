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

# Working with snapshots

Here, the following topics are going to be covered:

- What is a snapshot
- How to create it
- How it is saved next to the measurement data
- How to extract snapshot from the dataset

+++

### Useful imports

```{code-cell} ipython3
from pprint import pprint  # for pretty-printing python variables like 'dict'
import json  # for converting JSON data into python 'dict'

import qcodes
from qcodes import Parameter, Station, \
                   initialise_database, \
                   load_or_create_experiment, Measurement
from qcodes.tests.instrument_mocks import DummyInstrument
```

## What is a snapshot?

Often times experiments comprise a complex network of interconnected instruments. Their numerous settings define the overall behavior of the experimental setup. Obviously, the experimental setup has a direct impact on the measured data which is of prime interest for researchers. In order to capture this link, the measured data should have metadata associated with it. An important part of that metadata is a captured state of the experimental setup. In QCoDeS terms, this is called snapshot.

+++

## How to create a snapshot?

All QCoDeS instruments and parameters (and some other objects too, like `InstrumentChannel`s) support snapshotting, which means that they provide a method to retrieve their state.

Let's look at snapshots of various objects.

+++

### Snapshot example for Parameter object

+++

Let's create a `Parameter`, call its `snapshot` method, and then inspect the output of that method.

The returned snapshot is a python dictionary that reflects all the important properties of that parameter (check the name, label, unit, and even value).

```{code-cell} ipython3
p = Parameter('p', label='Parameter P', unit='kg', set_cmd=None, get_cmd=None)
p.set(123)

snapshot_of_p = p.snapshot()

pprint(snapshot_of_p)
```

In case you want to use the snapshot object in your code, you can refer to its contents in the same way as you work with python dictionaries, for example:

```{code-cell} ipython3
print(f"Value of {snapshot_of_p['label']} was {snapshot_of_p['value']} (when it was snapshotted).")
```

Note that the implementation of a particular QCoDeS object defines which attributes are snapshotted and how. For example, `Parameter` implements a keyword argument `snapshot_value` which allows to choose if the value of the parameter is snapshotted (the reasons for this are out of scope of this article). (Another interesting keyword argument of `Parameter` that is realated to snapshotting is `snapshot_get` - refer to `Parameters`'s docstring for more information.)

+++

Below is a demonstration of the `snapshot_value` keyword argument, notice that the value of the parameter is not part of the snapshot.

```{code-cell} ipython3
q = Parameter('q', label='Parameter Q', unit='A', snapshot_value=False, set_cmd=None, get_cmd=None)
p.set(456)

snapshot_of_q = q.snapshot()

pprint(snapshot_of_q)
```

### Snapshot of an Instrument

Now let's have a brief look at snapshots of instruments. For the sake of exercise, we are going to use a "dummy" instrument.

```{code-cell} ipython3
# A dummy instrument with two parameters, "input" and "output", plus a third one we'll use later.
instr = DummyInstrument('instr', gates=['input', 'output', 'gain'])
instr.gain(11)
```

```{code-cell} ipython3
snapshot_of_instr = instr.snapshot()

pprint(snapshot_of_instr, indent=4)
```

### Station and its snapshot

Experimental setups are large, and instruments tend to be quite complex in that they comprise many parameters and other stateful parts. It would be very time-consuming for the user to manually go through every instrument and parameter, and collect the snapshot data.

Here is where the concept of station comes into play. Instruments, parameters, and other submodules can be added to a [Station object](../Station.ipynb) ([nbviewer.jupyter.org link](https://nbviewer.jupyter.org/github/QCoDeS/Qcodes/tree/master/docs/examples/Station.ipynb)). In turn, the station has its `snapshot` method that allows to create a collective, single snapshot of all the instruments, parameters, and submodules.

Note that in this article the focus is on the snapshot feature of the QCoDeS `Station`, while it has some other features (also some legacy once).

+++

Let's create a station, and add a parameter, instrument, and submodule to it. Then we will print the snapshot. Notice that the station is aware of insturments and stand-alone parameters, and classifies them into dedicated lists within the snapshot.

```{code-cell} ipython3
station = Station()

station.add_component(p)
station.add_component(instr)
# Note that it is also possible to add components
# to a station via arguments of its constructor, like this:
#     station = Station(p, instr)
```

```{code-cell} ipython3
snapshot_of_station = station.snapshot()

pprint(snapshot_of_station)
```

## Saving snapshot next to the measurement data

With the power of the station object, it is now possible to conveniently associate the snapshot information with the measured data.

In order to do so, a station needs to be created, and then that station needs to be provided to the `Measurement` object. If no station is explicitly provided, the `Measurement` object will use the default station, `Station.default` (refer to `Measurement` and `Station` objects docstrings for more information). At the moment the new measurement run is started, a snapshot of the whole station will be taken, and added next to the measured data.

Note that the snapshot gets stored in a JSON format (an automatic convertion from python dictionary to JSON takes place). This is done in order to ensure that the snapshot can be read in environments other than python. JSON is an extemely popular data format, and all platforms/environments/languages/frameworks have means to read JSON-formatted data.

+++

Here is how it looks in the code. We will create a new experiment. Then we are going to reuse the station object created above, and create a new measurement object. Then we will perform a dummy measurement. After that we are going to extract the snapshot from the resulting dataset, and print it.

```{code-cell} ipython3
# Let's initialize a database to ensure that it exists
initialise_database()

# Let's create a new experiment
experiment = load_or_create_experiment('snapshot_experiment', 'no_sample_yet')
```

```{code-cell} ipython3
measurement = Measurement(experiment, station)

measurement.register_parameter(instr.input)
measurement.register_parameter(instr.output, setpoints=[instr.input])
```

```{code-cell} ipython3
with measurement.run() as data_saver:
    input_value = 111
    instr.input.set(input_value)

    instr.output.set(222)  # assuming that the instrument measured this value on the output

    data_saver.add_result((instr.input, input_value),
                          (instr.output, instr.output()))

# For convenience, let's work with the dataset object directly
dataset = data_saver.dataset
```

## Extracting snapshot from dataset

+++

Now we have a dataset that contains data from the measurement run. It also contains the snapshot.

In order to access the snapshot, use the `DataSet`'s properties called `snapshot` and `snapshot_raw`. As their docstrings declare, the former returns the snapshot of the run as a python dictionary, while the latter returns it as JSON string (in other words, in exactly the same format as it is stored in the experiments database).

```{code-cell} ipython3
snapshot_of_run = dataset.snapshot
```

```{code-cell} ipython3
snapshot_of_run_in_json_format = dataset.snapshot_raw
```

To prove that these snapshots are the same, use `json.loads` or `json.dumps` to assert the values of the variables:

```{code-cell} ipython3
assert json.loads(snapshot_of_run_in_json_format) == snapshot_of_run
```

```{code-cell} ipython3
assert json.dumps(snapshot_of_run) == snapshot_of_run_in_json_format
```

Finally, let's pretty-print the snapshot. Notice that the values of the `input` and `output` parameters of the `instr` instrument have `0`s as values, and not `111` and `222` that were set during the measurement run.

```{code-cell} ipython3
pprint(snapshot_of_run)
```

Note that the snapshot that we have just loaded from the dataset is almost the same as the snapshot that we directly obtained from the station above. The only difference is that the snapshot loaded from the dataset has a top-level `station` field. If you do not trust me, have a look at the following `assert` statement for the proof.

```{code-cell} ipython3
assert {'station': snapshot_of_station} == snapshot_of_run
```

## Comparing how two DataSets were taken

+++

Suppose something went wrong in an experiment, and you'd like to compare what changed since a known-good run.
QCoDeS lets you do this by taking a *diff* between the snapshots for two `DataSet` instances.

```{code-cell} ipython3
measurement = Measurement(experiment, station)

measurement.register_parameter(instr.input)
measurement.register_parameter(instr.output, setpoints=[instr.input])
```

```{code-cell} ipython3
instr.gain(400) # Oops!
with measurement.run() as data_saver:
    input_value = 111
    instr.input.set(input_value)

    instr.output.set(222)  # assuming that the instrument measured this value on the output

    data_saver.add_result((instr.input, input_value),
                          (instr.output, instr.output()))

# For convenience, let's work with the dataset object directly
bad_dataset = data_saver.dataset
```

The `diff_param_values` function tells us about the parameters that changed betw

```{code-cell} ipython3
from qcodes.utils.metadata import diff_param_values
```

```{code-cell} ipython3
diff_param_values(dataset.snapshot, bad_dataset.snapshot).changed
```
