# Object Hierarchy

## Rough linkages:

In **bold** the containing class creates this object.
In *italics* the container just holds this object (or class) as a default for derivatives to use.
Normal text the container includes and uses this object

- Station
  - BaseInstrument: IPInstrument, VisaInstrument, MockInstrument
    - **Parameter**
      - Validator: Anything, Strings, Numbers, Ints, Enum, MultiType
      - **SweepValues**: SweepFixedValues, AdaptiveSweep
    - Function
      - Validator
  - *SweepStorage class*: MergedCSVStorage, ?
  - **Monitor**
  - MeasurementSet
    - StorageManager, Monitor
    - *SweepStorage class*
    - .sweep
      - SweepValues
      - **SweepStorage**
  - **StorageManager**
    - **StorageServer**
      - SweepStorage
      - Monitor

## Station

A representation of the entire physical setup.

Lists all the connected `Instrument`s, keeps references to the `StorageManager` and
`Monitor` objects that are running, and holds various default settings, such as
the default `MeasurementSet` and the default `SweepStorage` class in use right now.

## Instrument

A representation of one particular piece of hardware.

Lists all `Parameter`s and `Function`s this hardware is capable of, and handles the
underlying communication with the hardware.
`BaseInstrument` sets the common structure but real instruments will generally derive
from its subclasses, such as `IPInstrument` and `VisaInstrument`. There is also
`MockInstrument` for making simulated instruments, connected to a `Model` that mimics
a serialized communication channel with an apparatus.

## Parameter

A representation of one particular state variable.

Most `Parameter`s are part of an `Instrument`, but you can also create `Parameter`s
that execute arbitrary functions, for example to combine several gate voltages in a
diagonal sweep. Parameters can have setters and/or getters (they must define at least
a setter OR a getter but do not need to define both)

`Parameter`s can be sliced to create a `SweepFixedValues` object.

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

## StorageManager

The gateway to the separate storage process.

Sweeps and monitoring routines all pass their data through the `StorageManager` to the
`StorageServer`, which the `StorageManager` started and is running in a separate process.
Likewise, plotting & display routines query the `StorageServer` via the `StorageManager`
to retrieve current data.

## StorageServer

Running in its own process, receives, holds, and returns current sweep and monitor data,
and writes it to disk (or other storage)

When a sweep is *not* running, the StorageServer also calls the monitor routine itself.
But when a sweep *is* running, the sweep calls the monitor so that it can avoid conflicts.
Also while a sweep is running, there are complementary `SweepStorage` objects in the sweep
and `StorageServer` processes - they are nearly identical objects, but are configured
differently.

## Monitor

Measures all system parameters periodically and logs these measurements

Not yet implemented, so I don't know the details, but the plan is that during idle times
it measures parameters one-by-one, keeping track of how long it took to measure each one,
so that during sweeps it can be given a command "measure for no longer than X seconds" and
it will figure out which parameters it has time to measure, and which of those are most
important (primarily, I guess, the most important were last measured the longest time ago)

## MeasurementSet

A list of parameters to measure while sweeping, along with the sweep methods.

Usually created from a `Station` and inheriting the `StorageManager`, `Monitor`,
and default `SweepStorage` class from it.

The `.sweep` method starts a sweep, which creates a `SweepStorage` object and,
if the sweep is to run in the background (the default), this method starts the
sweep process.

## SweepStorage

Object that holds sweep data and knows how to read from and write to disk or
other long-term storage, as well as how to communicate with its clone on the
`StorageServer` if one exists.

A `SweepStorage` is created when a new sweep is started, and this one clones
itself on the `StorageServer`. But you can also create a `SweepStorage` from
a previous sweep, in which case it simply reads in the sweep and holds it
for plotting or analysis.

Subclasses (eg `MergedCSVStorage`, later perhaps `AzureStorage` etc) define
the connection with the particular long-term storage you are using.
