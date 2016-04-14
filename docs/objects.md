# Object Hierarchy

## Rough linkages:

In **bold** the containing class creates this object.
In *italics* the container just holds this object (or class) as a default for derivatives to use.
Normal text the container includes and uses this object

- Station
  - Instrument: IPInstrument, VisaInstrument, MockInstrument
    - **Parameter**: StandardParameter
      - Validator: Anything, Strings, Numbers, Ints, Enum, MultiType
      - **SweepValues**: SweepFixedValues, AdaptiveSweep
    - Function
      - Validator
  - **Monitor**
  - *actions*
- DataManager
  - **DataServer**
- Loop
  - actions: Parameter, Task, Wait, (Active)Loop
  - **ActiveLoop**
    - **DataSet**
      - **DataArray**
      - **Formatter**: GNUPlotFormat
      - **DiskIO** (may become subclass of IOManager?)
      - **TimestampLocation** (a location_provider)

## Station

A representation of the entire physical setup.

Lists all the connected `Instrument`s and the current default
measurement (a list of actions). Contains a convenience method
`.measure()` to measure these defaults right now, but this is separate
from the code used by `Loop`.

## Instrument

A representation of one particular piece of hardware.

Lists all `Parameter`s and `Function`s this hardware is capable of, and handles the
underlying communication with the hardware.
`Instrument` sets the common structure but real instruments will generally derive
from its subclasses, such as `IPInstrument` and `VisaInstrument`. There is also
`MockInstrument` for making simulated instruments, connected to a `Model` that mimics
a serialized communication channel with an apparatus.

## Parameter

A representation of one particular state variable.

Most `Parameter`s are part of an `Instrument`, using the subclass
`StandardParameter` which links it to specific commands sent to a specific
instrument. But you can also create `Parameter`s that execute arbitrary functions,
for example to combine several gate voltages in a diagonal sweep. Parameters can
have setters and/or getters (they must define at least a setter OR a getter but
do not need to define both)

Gettable `Parameter`s often return one value, but can return a collection of
values - several values with different names, or one or more arrays of values,
ie if an instrument provides a whole data set at once, or if you want to save
raw and calculated data side-by-side

Settable `Parameter`s take a single value, and can be sliced to create a
`SweepFixedValues` object.

## Validator

Defines a set of valid inputs, and tests values against this set

Subclasses include `Anything`, `Strings`, `Numbers`, `Ints`, `Enum`, and `MultiType`,
each of which takes various arguments that restrict it further.

## SweepValues

An iterator that provides values to a sweep, along with a setter for those values
connected to a `Parameter`.

Mostly the `SweepFixedValues` subclass is used (this is
flexible enough to iterate over any arbitrary collection of values, not necessarily
linearly spaced) but other subclasses can execute adaptive sampling techniques, based
on the `.feedback` method by which the sweep passes measured values back into the
`SweepValues` object.

## Function

A representation of some action an `Instrument` can perform, which is not connected to
any particular state variable (eg calibrate, reset, identify)

## Monitor

Measures all system parameters periodically and logs these measurements

Not yet implemented, so I don't know the details, but the plan is that during idle times
it measures parameters one-by-one, keeping track of how long it took to measure each one,
so that during sweeps it can be given a command "measure for no longer than X seconds" and
it will figure out which parameters it has time to measure, and which of those are most
important (primarily, I guess, the most important were last measured the longest time ago)

## DataManager

The gateway to the separate DataServer process.

Measurement `Loop`s and monitoring routines all pass their data through the
`DataManager` to the `DataServer`, which the `DataManager` started and is
running in a separate process.
Likewise, plotting & display routines query the `DataServer` via the `DataManager`
to retrieve current data.

## DataServer

Running in its own process, receives, holds, and returns current `Loop` and
monitor data, and writes it to disk (or other storage)

When a `Loop` is *not* running, the DataServer also calls the monitor routine.
But when a `Loop` *is* running, *it* calls the monitor so that it can avoid conflicts.
Also while a `Loop` is running, there are complementary `DataSet` objects in the loop
and `DataServer` processes - they are nearly identical objects, but are configured
differently so that the loop `DataSet` doesn't hold any data itself, it only
passes that data on to the `DataServer`

## Loop

Describes a sequence of `Parameter` settings to loop over. When you attach
`action`s to a `Loop`, it becomes an `ActiveLoop` that you can `.run()`, or
you can run a `Loop` directly, in which case it takes the default `action`s from
the default `Station`

`actions` are a sequence of things to do at each `Loop` step: they can be
`Parameter`s to measure, `Task`s to do (any callable that does not yield data),
`Wait` times, other `ActiveLoop`s to nest inside this one, or `Loop`s to nest using
the default `actions`.

## ActiveLoop

After a `Loop` has actions attached to it, all you can do with it is `.run()` it,
or use it as an action inside another `Loop` which will `.run()` the whole thing.

The `ActiveLoop` determines what `DataArray`s it will need to hold the data it
collects, and it creates a `DataSet` holding these `DataArray`s

## DataSet

A collection of `DataArray`s that contains all the data from one measurement
(normally one `Loop`). During measurement, it creates three copies of itself: one
in the loop process that just passes data to the `DataServer`, one on the `DataServer`
that holds the data and stores it to disk, one in the main thread that can sync to
the one in the `DataServer` on demand for live plotting and analysis.

You also get a `DataSet` if you read back in an old data set, so it can use the
same API for plotting and analysis during and after acquisition.

Storage of a `DataSet` is described by three objects - an IO manager, a Location Provider,
and a Formatter. Their default values are stored in class attributes:
- `DataSet.default_io = DiskIO('.')`
- `DataSet.location_provider = TimestampLocation()`
- `DataSet.default_formatter = GNUPlotFormat()`.

All of these can be changed for ALL `DataSet`s you create or load by setting these class
attributes in an init script (ie `DataSet.default_io = DiskIO('c:/path/to/data')`) or as
kwargs to `Loop.run` or `load_data` ie:
```
Loop.run(io=DiskIO(...),
         location=<location_provider callable or location string>,
         formatter=HDF5Format(...))
```

## DataArray

One n-dimensional data array. If it is a measured `Parameter`, it references
other `DataArray`(s) as its setpoints.

## Formatter

Describes the file format used to read and write a `DataSet`. Uses an
IO Manager to actually perform the storage (so that the same format can be
used on disk, in the cloud, in a database...)

## IO Manager

Performs the actual reading and writing of files generated by a `Formatter`.
Only one IO manager has been implemented so far, `DiskIO`, and the default base
location is `'.'`, ie the directory in which you started you python session.
If you want to store data on a cloud service, or over FTP, or something like that, 
make another IO manager class that defines the same interface as `DiskIO`.

## Location Provider

A callable that returns a string that identifies a unique location as understood by
IO managers. For `DiskIO` this string is interpreted as a path relative to its
base location. The callable takes two parameters:
- the IO manager `io`, so that it can call `io.list(location)` to see if there's
already something stored there and increment a counter to make a new location
- an optional `name` (you can call `Loop.run(name='first_electron')` to provide this)
that gets incorporated into the path

One Location Provider class has been implemented, `TimestampLocation`, which can be
instantiated with a format (`datetime.strftime` format string) that defaults to
`'%Y-%m-%d/%H-%M-%S'` ie a folder per day, and inside that a filename, or another
folder, depending on how the Formatter is implemented.
