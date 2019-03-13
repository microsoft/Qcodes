.. _introduction:

Introduction
============

.. toctree::
   :maxdepth: 2


The framework is designed for extreme modularity. Don't panic when looking at the source code.
Read the following descriptions of the *pillars* of the framework first.


Overview
--------

A QCoDeS experiment typically consists of a Loop_ that sweeps over one or more Parameters_ of one or more Instruments_,
measures other Parameters_ at each sweep point, and stores all of the results into a DataSet_.

.. _Parameters: Parameter_
.. _Instruments: Instrument_

While the simple case is quite straightforward, it is possible to create a very general experiment by defining richer Parameters and 
by performing additional Loop actions at each sweep point. 
The overview on this page provides a high-level picture of the general capabilities;
consult the detailed API references and the samples to see some of the complex procedures that can be described and run.

Instrument
----------
.. Description

An instrument is first and most fundamental pillar of qcodes as it represent the hardware you would want to talk to, either to control your system, collect data, or both.

Instruments come in several flavors:
  - Hardware: most instruments map one-to-one to a real piece of hardware; in these instances, the QCoDeS Instrument requires a driver or communication channel to the hardware. See  :ref:`driver`.

  - Simulation: for theoretical or computational work, an instrument may contain or connect to a model that has its own state and generates results in much the same way that an instrument returns measurements. See :ref:`simulation`.

  - Manual: If a real instrument has no computer interface (just physical switches and knobs), you may want a QCoDeS instrument for it just to maintain a record of how it was configured at any given time during the experiment, and perhaps to use those settings in calculations. In these cases it is of course up to the user to keep the hardware and software synchronized.

  - Meta: Sometimes to make the experiment easier to manage, it is useful to make an instrument to represent some element of the system that may be controlled in part by several separate instruments. For example, a device that uses one instrument to supply DC voltages, another to supply AC signals, and a third to measure it. We can make a new QCoDeS Instrument that references these lower-level Instruments, and we refer to this as a "Meta-Instrument". This imposes some requirements and limitations with remote instruments and multiple processes - see :ref:`metainstrument` for more information.

An instrument can exist as local instrument or remote instrument. A local instrument is instantiated in the main process, and you interact with it directly. This is convenient for testing and debugging, but a local instrument cannot be used with a measurement loop in a separate process (ie a background loop). For that purpose you need a remote instrument. A remote instrument starts (or connects to) a server process, instantiates the instrument in that process, and returns a proxy object that mimicks the API of the instrument. This proxy holds no state or hardware connection, so it can be freely copied to other processes, in particular background loops and composite-instrument servers.

.. responsibilities

Instruments are responsible for:
  - Holding connections to hardware, be it VISA, some other communication protocol, or a specific DLL or lower-level driver.
  - Creating a parameter_, ref:`function`, or method for each piece of functionality we support. These objects may be used independent of the instrument, but they make use of the instrument's hardware connection in order to do their jobs.
  - Describing their complete current state ("snapshot") when asked, as a JSON-compatible dictionary.

.. state

Instruments hold state of:
  - The communication address, and in many cases the open communication channel.
  - A list of references to parameters added to the instrument.

.. failures

Instruments can fail:
  - When a VisaInstrument has been instantiated before, particularly with TCPIP, sometimes it will complain "VI_ERROR_RSRC_NFOUND: Insufficient location information or the requested device or resource is not present in the system" and not allow you to open the instrument until either the hardware has been power cycled or the network cable disconnected and reconnected. Are we using visa/pyvisa in a brittle way?
  - If you try to use a background loop with a local instrument, because that would require copying the local instrument and there may only be one local copy of the instrument (if you make a remote instrument, the server instance is the one local copy).


Parameter
---------

.. Description

A Parameter represents a state variable of your system. 
Parameters may be settable, gettable, or both. 
While many Parameters represent a setting or measurement for a particular Instrument, 
it is possible to define Parameters that represent more powerful abstractions.

The variable represented by a Parameter may be a simple number or string.
It may also be a complicated data structure that contains numerical, textual, or other information.
More information is available in the API documentation on the Parameter type.

Responsibilities
~~~~~~~~~~~~~~~~

.. responsibilities

Parameters are responsible for:
  - (if part of an Instrument) generating the commands to pass to the Instrument and interpreting its response
  - (if not part of an Instrument) providing get and/or set methods
  - (if settable) testing whether an input is valid, usually using a :mod:`validator. <qcodes.utils.validators>`
  - providing context and meaning to the parameter through descriptive attributes:
    - name: its name as an attribute
    - label: short description as in an axis label
    - units: if the values are numbers, their units
    - and more if multi-valued or array-valued

.. state

Parameters hold onto their latest set or measured value, as well as the timestamp of the latest update.
Thus, snapshots need not always query the hardware for this information, but can update it intelligently when it has gotten stale.

.. failures

A Parameter that is part of an Instrument, even though it can be used as an independent object without directly referencing the Instrument, 
is subject to the same local/remote limitations as the Instrument.

Examples
~~~~~~~~

We list some common types of Parameters here:

**Instrument Parameters**

The simplest Parameters are part of an Instrument_.
These Parameters are created using ``instrument.add_parameter()`` and use the Instrument's low-level communication methods for execution.

A settable Parameter typically represents a configuration setting or other controlled characteristic of the Instrument. 
Most such Parameters have a simple numeric value, but the value can be a string or other data type if necessary.
If a settable Parameter is also gettable, getting it typically just reads back what was previously set, through QCoDeS or by some other means,
but there can be differences due to rounding, clipping, feedback loops, etc. 
Note that setting a Parameter of a :ref:`metainstrument` may involve setting several lower-level Parameters of the underlying Instruments, 
or even getting the values of other Parameters to inform the value(s) to set.

A Parameter that is only gettable typically represents a single measurement command or sequence.

The value of such a Parameter may be of many types:
  - A single numeric value, such as a voltage measurement
  - A string that represents a discrete instrument setting, such as the orientation of a vector
  - Multiple related values, such as the magnitude and phase or Cartesian components of a vector
  - A sequence of values, such as a sampled waveform or a power spectrum
  - Multiple sequences of values, such as waveforms sampled on multiple channels
  - Any other shape that appropriately represents a characteristic of the Instrument.

When a RemoteInstrument is created, the Parameters contained in the Instrument are mirrored as RemoteParameters, 
which connect to the original Parameter via the associated InstrumentServer.

**Computed Measurements**

In some cases the measurement value of interest is computed based on values read from more than one instrument.
For example, you might want to track the power dissipation of a component, computed by multiplying the outputs from a voltmeter and an ammeter.
QCoDeS allows you to define a Parameter that represents the result of such a computation and include it as part of your experimental results.

A Parameter defined in this way is not associated with an Instrument.
It is a Python object that encapsulates the computation to be performed and references the underlying Parameters it uses.
It is gettable, but not settable.

**Interdependent Settings**

In some experiments, control values for one or more Instruments must be set together in order to maintain a condition.
For example, you might want to measure the behavior of a component at different voltage levels, 
but always keeping the available power within a fixed bound.
You can define a Parameter that gets initialized with the maximum power to allow, such that setting the Parameter value
results in setting the voltage to the passed-in value and also adjusting the supplied current appropriately.

Similarly to a Parameter that computes a measurement result, this type of Parameter is not associated with an Instrument.
It encapsulates the way that the single setting impacts the related settings and references the underlying Parameters, one for each setting.
It is settable, and may be gettable.


Loop
----

.. Description

A Loop is the QCoDeS way to acquire one or more arrays of data. Every Loop that's executed consists of a settable Parameter to be varied, some collection of values to set it to, some actions to do at each setpoint, and some conditions by which to run the Loop.

An action can be:
  - A gettable Parameter (something to measure). Each such Parameter will generate one (or more, if the Parameter itself creates multiple outputs).
  - A Task to do (for example you measure once, then have a Task to change a gate voltage, then you measure again, and finally a Task to put the gate voltage back where it was).
  - Wait, a specialized task that just delays execution (but may do other things like monitoring the system in that time)
  - BreakIf, a callable test for whether to quit (this level of) the Loop.
  - Another Loop nested inside the first, with all its own setpoints and actions. Measurements within a nested loop will produce a higher-dimension output array.

The key loop running conditions are:
  - background or not: A background Loop runs in its own separate process, so that you can be doing things in the main process like live plotting, analysis on the data as it arrives, preparing for the next measurement, or even unrelated tasks, while the Loop is running. The disadvantage is complexity, in that you can only use RemoteInstruments, and debugging gets much harder.
  - use threads: If true, we will group measurement actions and try to execute them concurrently across several threads. This can dramatically speed up slow measurements involving several instruments, but only if all instruments are in separate InstrumentServer processes, or all instruments are local.
  - data manager: If not False, we create another extra process whose job it is to offload data storage, and sync data back to the main process on demand, so that the Loop process can run with as little overhead as possible.
  - where and how to save the data to disk

.. responsibilities

The Loop is responsible for:
  - creating the dataset_ that will be needed to store its data
  - generating all the metadata for the DataSet. Metadata is intended to describe the system and software configuration to give it context, help reproduce and troubleshoot the experiment, and to aid searching and datamining later. The Loop generates its own metadata, regarding when and how it was run and the Parameters and other actions involved, as well as asking all the Instruments, via a :ref:`station_api` if possible, for their own metadata and including it.
  - sequencing actions: the Loop should have the highest priority and the least overhead of extra responsibilities so that setpoints and actions occur with as fast and reliable timing as possible.

.. state

Before the Loop is run, it holds the setpoint and action definitions you are building up. You can actually keep a loop at any level of definition and reuse it later. Loop methods chain by creating entirely new objects, so that you can hold onto the Loop at any stage of definition and reuse just what has been defined up to that point.

After the Loop is run, it returns a dataset_ and the executed loop itself, along with the process it starts if it's a background Loop, only hold state (such as the current indices within the potentially nested Loops) while it is running.

.. failures

Loops can fail:
  - If you try to use a (parameter of a) local instrument in a background loop

Measure
-------

.. Description

If you want to create a dataset_ without running a loop_ - for example, from a single Parameter.get() that returns one or more whole arrays - you can use Measure. Measure works very similarly to Loop, accepting all the same action types. The API for running a Measure is also very similar to Loop, with the difference that Measure does not allow background acquisition.

If any of the actions return scalars, these will be entered in the DataSet as 1D length-1 arrays, along with a similar length-1 setpoint array.

.. responsibilities

All the same as a Loop

.. state

Just like a Loop, you can hold a Measure object, with its list of actions to execute, and reuse it multiple times.

DataSet
-------

.. Description

A DataSet is a way to group arrays of data together, describe the meaning of each and their relationships to each other, and record metadata.

Typically a DataSet is the result of running a single Loop, and contains all the data generated by the Loop as well as all the metadata necessary to understand and repeat it.

The data in a DataSet is stored in one or more :mod:`DataArray <qcodes.data.data_array.DataArray>` objects, each of which is a single numpy ndarray (wrapped with some extra functionality and attributes). The metadata is stored in a JSON-compatible dictionary structure.

A DataArray with N dimensions should list N setpoint arrays, each of which is also a DataArray in the same DataSet. The first setpoint array should have 1 dimension, the second 2 dimensions, etc. This follows the procedure of most experimental loops, where the outer loop parameter only changes when you increment the outer loop.

If your loop does *not* work this way, and the setpoint of the first index changes with the second index, you should either use an array of integers as the outer setpoints, and treat your varying indices as a separate measured array, or you may prefer to store all of the setpoints and measurements as 1D arrays, where each index represents one condition across all arrays, akin to an SQL table (where each array would represent one column of the table).

One DataArray can only be part of at most one DataSet. This ensures that we don't generate irreversible situations by saving an array in multiple places and reloading them separately, or conflicts if we try to sync (or reload) several DataSets with inconsistent data in the multiply-referenced arrays, and that we can always refer from the DataArray to a single DataSet, which is important for live plotting.

The DataSet also specifies where and how it is to be stored on disk. Storage is specified by an io_manager (the physical device / protocol, and base location in the normal case of disk storage), a location (string, relative path within the io manager), and :mod:`formatter <qcodes.data.format.Formatter>` (specifies the file type and how to read to and write from a DataSet).

A DataManager can be used during acquisition to offload io operations to a separate DataServer process, and to update any live plots without burdening the Loop process.

A DataSet can have one of three modes (in reference to the DataServer):
  - PUSH_TO_SERVER: this DataSet and its DataArrays maintain no local copy of the data. Whenever new data arrives it is sent immediately to the DataServer. At the end of a background loop, this object just disappears. At the end of a foreground loop, it asks to sync with the server, at which point it's done so it becomes LOCAL.
  - PULL_FROM_SERVER: updates are requested from the DataServer when the sync() method is called (eg during live plotting) such that after updating, this copy contains all the same data as on the server. When the DataServer indicates that the DataSet is done, the mode changes to LOCAL.
  - LOCAL: No communication with the server. If no server is specified, this is the only allowed mode.

.. note:: metadata is not (yet) integrated with the DataServer. It is (almost) entirely dealt with locally: in the main process before the Loop process (if any) starts, the station snapshot and loop definition is recorded and saved to disk. The only thing that happens later is the loop end timestamp gets added immediately before the Loop terminates

.. responsibilities

The DataSet is responsible for:
  - Accepting incremental pieces of data (setpoints and measurements as they become available)
  - Either holding that data locally (within its DataArrays), or pushing it to another copy of itself that stores it
  - If it's a copy that holds data, each DataArray maintains a record of the range of indices that have changed since the last save to storage, the last index that *has* been saved, and (if it's in PULL_FROM_SERVER mode) the last index that has been synced from the server. This implicitly assumes that the DataArrays are filled in order of the raveled indices, ie looping over the inner index first.
  - It's up to the Formatter to look at each of these DataArrays, decide what parts of the changes in each to save to storage, and then tell each DataArray what it saved (expressed as a greatest raveled index). With that information the DataArray updates its record of what still needs saving. This is done so that a Formatter can choose to combine several DataArrays into one table, which may require writing only the values at array positions which have been finished in all of these arrays.

.. state

Each DataSet holds:
  - Its own metadata (JSON-compatible dict)
  - Its mode (PUSH_TO_SERVER, PULL_FROM_SERVER, LOCAL)
  - A dict of DataArrays, each with attributes: name (which is also its dictionary key in DataSet.arrays), label, units, setpoints. If the DataSet is in PUSH_TO_SERVER mode, these DataArrays do not hold any data. Otherwise, these DataArrays contain numpy arrays of data, as well as records (as described above) of what parts of that array have been changed, saved, and synced.
  - location, formatter, and io manager

.. failures

DataSets can fail:
  - If somehow the data in storage does not match the record in memory of what it has saved, for example if you change the stored file during acquisition. The consequences depend on the formatter (this could be completely destructive for GNUPlotFormat or other text-based formats, probably less so for HDF5) but in general the DataSet has no way of independently checking that the existing data on disk is still what it thinks it is. A safe but slow way around this is to rewrite the stored files completely

  - All combinations of background/foreground loops and yes/no DataManager are supported, but there is at least one caveat: if you use a foreground loop (for example you want to not use instrument servers) with a DataManager (you still want to offload the IO to a separate process for performance and regularity in the main/loop process) you will not be able to use a separate thread to do live plotting from within the main process, because the copy of the DataSet there will not contain any data! After the loop has finished, plotting should work fine in the main process, because at the end of the loop the DataSet will be converted to PULL_FROM_SERVER mode and the data synced back from the server to the main process.

  - if the location, formatter, or io manager is changed manually and then you try to write the DataSet, it *should* figure out that there are really no files there and rewrite the whole thing, but it's better to use write_copy, which explicitly marks the DataSet as unsaved, then saves it in the new location. We probably want to wrap these attributes in ``@property`` calls that mark the data as unsaved if you change any of them.
