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

# Legacy features
This section describes legacy features from early implementations of QCoDeS. As new features often use similar (or identical) names and achieve similar goals, these older implementations have been laid to rest in these example notebooks. The intent of preserving this data is twofold: first to preserve documentation for users still relient on these systems, and second to better understand these older implementations should their code be revisted.

In addition to the information here, we aim to provide links to sections of documentation where updated information can be found, this is intended to assist early users in transitioning to newer features.

## Overview
A QCoDeS experiment typically consists of a Loop that sweeps over one or more Parameters of one or more Instruments, measures other Parameters at each sweep point, and stores all of the results into a DataSet.

While the simple case is quite straightforward, it is possible to create a very general experiment by defining richer Parameters and by performing additional Loop actions at each sweep point. The overview on this page provides a high-level picture of the general capabilities; consult the detailed API references and the samples to see some of the complex procedures that can be described and run.


## Loop
A Loop is the QCoDeS way to acquire one or more arrays of data. Every Loop that’s executed consists of a settable Parameter to be varied, some collection of values to set it to, some actions to do at each setpoint, and some conditions by which to run the Loop.

An action can be:
- A gettable Parameter (something to measure). Each such Parameter will generate one (or more, if the Parameter itself creates multiple outputs).
- A Task to do (for example you measure once, then have a Task to change a gate voltage, then you measure again, and finally a Task to put the gate voltage back where it was).
- Wait, a specialized task that just delays execution (but may do other things like monitoring the system in that time)
- BreakIf, a callable test for whether to quit (this level of) the Loop.

Another Loop nested inside the first, with all its own setpoints and actions. Measurements within a nested loop will produce a higher-dimension output array.

The key loop running conditions are:
- background or not: A background Loop runs in its own separate process, so that you can be doing things in the main process like live plotting, analysis on the data as it arrives, preparing for the next measurement, or even unrelated tasks, while the Loop is running. The disadvantage is complexity, in that you can only use RemoteInstruments, and debugging gets much harder.
- use threads: If true, we will group measurement actions and try to execute them concurrently across several threads. This can dramatically speed up slow measurements involving several instruments, or all instruments are local.
- data manager: If not False, we create another extra process whose job it is to offload data storage, and sync data back to the main process on demand, so that the Loop process can run with as little overhead as possible.


### Responsibilities:

- creating the dataset that will be needed to store its data
- where and how to save the data to disk
- generating all the metadata for the DataSet.

> Metadata is intended to describe the system and software configuration to give it context, help reproduce and troubleshoot the experiment, and to aid searching and datamining later. The Loop generates its own metadata, regarding when and how it was run and the Parameters and other actions involved, as well as asking all the Instruments, via a qcodes.station if possible, for their own metadata and including it.

- sequencing actions: the Loop should have the highest priority and the least overhead of extra responsibilities so that setpoints and actions occur with as fast and reliable timing as possible.

Before the Loop is run, it holds the setpoint and action definitions you are building up. You can actually keep a loop at any level of definition and reuse it later. Loop methods chain by creating entirely new objects, so that you can hold onto the Loop at any stage of definition and reuse just what has been defined up to that point.

After the Loop is run, it returns a dataset and the executed loop itself, along with the process it starts if it’s a background Loop, only hold state (such as the current indices within the potentially nested Loops) while it is running.

### Loops can fail:
If you try to use a (parameter of a) local instrument in a background loop

## Measure
If you want to create a dataset without running a loop - for example, from a single Parameter.get() that returns one or more whole arrays - you can use Measure. Measure works very similarly to Loop, accepting all the same action types. The API for running a Measure is also very similar to Loop, with the difference that Measure does not allow background acquisition.

If any of the actions return scalars, these will be entered in the DataSet as 1D length-1 arrays, along with a similar length-1 setpoint array.

Just like a Loop, you can hold a Measure object, with its list of actions to execute, and reuse it multiple times.

## DataSet

A DataSet is a way to group arrays of data together, describe the meaning of each and their relationships to each other, and record metadata.

Typically a DataSet is the result of running a single Loop, and contains all the data generated by the Loop as well as all the metadata necessary to understand and repeat it.

The data in a DataSet is stored in one or more DataArray objects, each of which is a single numpy ndarray (wrapped with some extra functionality and attributes). The metadata is stored in a JSON-compatible dictionary structure.

A DataArray with N dimensions should list N setpoint arrays, each of which is also a DataArray in the same DataSet. The first setpoint array should have 1 dimension, the second 2 dimensions, etc. This follows the procedure of most experimental loops, where the outer loop parameter only changes when you increment the outer loop.

If your loop does not work this way, and the setpoint of the first index changes with the second index, you should either use an array of integers as the outer setpoints, and treat your varying indices as a separate measured array, or you may prefer to store all of the setpoints and measurements as 1D arrays, where each index represents one condition across all arrays, akin to an SQL table (where each array would represent one column of the table).

One DataArray can only be part of at most one DataSet. This ensures that we don’t generate irreversible situations by saving an array in multiple places and reloading them separately, or conflicts if we try to sync (or reload) several DataSets with inconsistent data in the multiply-referenced arrays, and that we can always refer from the DataArray to a single DataSet, which is important for live plotting.

The DataSet also specifies where and how it is to be stored on disk. Storage is specified by an io_manager (the physical device / protocol, and base location in the normal case of disk storage), a location (string, relative path within the io manager), and formatter (specifies the file type and how to read to and write from a DataSet).

### Responsibilities:

Accepting incremental pieces of data (setpoints and measurements as they become available)

Either holding that data locally (within its DataArrays), or pushing it to another copy of itself that stores it

If it’s a copy that holds data, each DataArray maintains a record of the range of indices that have changed since the last save to storage, the last index that has been saved, and (if it’s in PULL_FROM_SERVER mode) the last index that has been synced from the server. This implicitly assumes that the DataArrays are filled in order of the raveled indices, ie looping over the inner index first.

It’s up to the Formatter to look at each of these DataArrays, decide what parts of the changes in each to save to storage, and then tell each DataArray what it saved (expressed as a greatest raveled index). With that information the DataArray updates its record of what still needs saving. This is done so that a Formatter can choose to combine several DataArrays into one table, which may require writing only the values at array positions which have been finished in all of these arrays.

Each DataSet holds:

- Its own metadata (JSON-compatible dict, i.e., everything that the custom JSON encoder class qcodes.utils.NumpyJSONEncoder supports.)
- Its mode (PUSH_TO_SERVER, PULL_FROM_SERVER, LOCAL)
- A dict of DataArrays, each with attributes: name (which is also its dictionary key in DataSet.arrays), label, units, setpoints. If the DataSet is in PUSH_TO_SERVER mode, these DataArrays do not hold any data. Otherwise, these DataArrays contain numpy arrays of data, as well as records (as described above) of what parts of that array have been changed, saved, and synced.
- location, formatter, and io manager

### DataSets can fail:

If somehow the data in storage does not match the record in memory of what it has saved, for example if you change the stored file during acquisition. The consequences depend on the formatter (this could be completely destructive for GNUPlotFormat or other text-based formats, probably less so for HDF5) but in general the DataSet has no way of independently checking that the existing data on disk is still what it thinks it is. A safe but slow way around this is to rewrite the stored files completely