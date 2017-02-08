=====================
DataSet Specification
=====================

Introduction
============

The DataSet class is used in QCoDeS to hold measurement results.
It is the destination for measurement loops and the source for plotting and data analysis.
As such, it is a central component of QCoDeS.

The DataSet class should be usable on its own, without other QCoDeS components.
In particular, the DataSet class should not require the use of Loop and parameters, although it should integrate with those components seamlessly.
This will significantly improve the modularity of QCoDeS by allowing users to plug into and extend the package in many different ways.
As long as a DataSet is used for data storage, users can freely select the QCoDeS components they want to use.

Terminology
================

Parameter
    A logically-single value input to or produced by a measurement.
    A parameter need not be a scalar, but can be an array or a tuple or an array of tuples, etc.
    A DataSet parameter corresponds conceptually to a QCoDeS parameter, but does not have to be defined by or associated with a QCoDeS Parameter . 
    Roughly, a parameter represents a column in a table of experimental data.
    
Result
    A result is the collection of parameter values associated to a single measurement in an experiment.
    Roughly, a result corresponds to a row in a table of experimental data.
    
DataSet
    A DataSet is a QCoDeS object that stores the results of an experiment.
    Roughly, a DataSet corresponds to a table of experimental data, along with metadata that describes the data.
    Depending on the state of the experiment, a DataSet may be "in progress" or "completed".

ExperimentContainer
    An ExperimentContainer is a QCoDeS object that stores all information about an experiment.
    This includes items such as the equipment on which the experiment was run, the configuration of the equipment, graphs and other analytical output, and arbitrary notes, as well as the DataSet that holds the results of the experiment.

Requirements
============

The DataSet class should meet the following requirements:

Basics
---------

#. A DataSet can store data of (reasonably) arbitrary types and shapes. basically, any type and shape that can fit in a NumPy array should be supported.
#. The results stored in a completed DataSet should be immutable; no new results may be added to a completed DataSet.

Creation
------------

#. It should be possible to create a DataSet without knowing the final item count of the various values it stores. 
   In particular, the number of loop iterations for a sweep should not be required to create the DataSet.
#. The list of parameters in each result to be stored in a DataSet may be specified at creation time.
   This includes the name, role (set-point or output), and type of each parameter.
   Parameters may be marked as optional, in which case they are not required for each result.
#. It should be possible to add a new parameter to an in-progress DataSet.
#. It should be possible to define a result parameter that is independent of any QCoDeSParameter or Instrument.
#. A QCoDeS Parameter should provide sufficient information to define a result parameter.
#. A DataSet should allow storage of relatively arbitrary metadata describing the run that 
   generated the results and the parameters included in the results.
   Essentially, DataSet metadata should be a string-keyed dictionary at the top, 
   and should allow storage of any JSON-encodable data.
    
Writing
----------

#. It should be possible to add a single result or a sequence of results to an in-progress DataSet.
#. It should be able to add an array of values for a new parameter to an in-progress DataSet.
#. A DataSet should maintain the order in which results were added.
#. An in-progress DataSet may be marked as completed.

Access
---------

#. Values in a DataSet should be easily accessible for plotting and analysis, even while the DataSet is in progress.
   In particular, it should be possible to retrieve full or partial results as a NumPy array.
#. It should be possible to define a cursor that specifies a location in a specific value set in a DataSet.
   It should be possible to get a cursor that specifies the current end of the DataSet when the DataSet is "in progress".
   It should be possible to read "new data" in a DataSet; that is, to read everything after a cursor.
#. It should be possible to subscribe to change notifications from a DataSet.
   It is acceptable if such subscriptions must be in-process until QCoDeS multiprocessing is redone.

Storage and Persistence
-----------------------

#. A DataSet object should allow writing to and reading from storage in a variety of formats.
#. Users should be able to define new persistence formats.
#. Users should be able to specify where a DataSet is written.

Interface
=========

Creation
--------

ParamSpec
~~~~~~~~~

A ParamSpec object specifies a single parameter in a DataSet.

ParamSpec(name, type, metadata=)
    Creates a parameter specification with the given name and type. 
    The type should be a NumPy dtype object.
    
    If metadata is provided, it is included in the overall metadata of the DataSet, with the name of the parameter as the top-level tag.
    The metadata can be any JSON-able object.
	
Either the QCoDeS Parameter class should inherit from ParamSpec, or the Parameter class should provide
a simple way to get a ParamSpec for the Parameter.

DataSet
~~~~~~~

DataSet()
    Creates a DataSet with no parameters.

DataSet(specs)
    Creates a DataSet for the provided list of parameter specifications.
    Each item in the list should be a ParamSpec object.
	
DataSet(specs, values)
    Creates a DataSet for the provided list of parameter specifications and values.
    Each item in the specs list should be a ParamSpec object.
	Each item in the values list should be a NumPy array or a Python list of values for the corresponding ParamSpec.
	There should be exactly one item in the values list for every item in the specs list.
	All of the arrays/lists in the values list should have the same length.
	The values list may intermix NumPy arrays and Python lists.

DataSet.add_parameter(spec)
    Adds a parameter to the DataSet.
    The spec should be a ParamSpec object.
    If the DataSet is not empty, then existing results will have the type-appropriate null value for the new parameter.
    
    It is an error to add parameters to a completed DataSet.

DataSet.add_parameters(specs)
    Adds a list of parameters to the DataSet.
    Each item in the list should be a ParamSpec object.
    If the DataSet is not empty, then existing results will have the type-appropriate null value for the new parameters.
    
    It is an error to add parameters to a completed DataSet.

DataSet.add_metadata(tag=, metadata=)
    Adds metadata to the DataSet.
    The metadata is stored under the provided tag.
	If there is already metadata under the provided tag, the new metadata replaces the old metadata.
    The metadata can be any JSON-able object.

Writing
-------

DataSet.add_result(**kwargs)
    Adds a result to the DataSet.
    Keyword parameters should have the name of a parameter as the keyword and the value to associate as the value.
    If there is only one positional parameter and it is a dictionary, then it is interpreted as a map from parameter name to parameter value.
    
    It is an error to provide a value for a key or keyword that is not the name of a parameter in this DataSet.
    
    It is an error to add a result to a completed DataSet.

DataSet.add_results(args)
    Adds a sequence of results to the DataSet.
    The single argument should be a sequence of dictionaries, where each dictionary provides the values for all of the parameters in that result.
    See the add_result method for a description of such a dictionary.
    The order of dictionaries in the sequence will be the same as the order in which they are added to the DataSet.
    
    It is an error to add results to a completed DataSet.

DataSet.add_parameter_values(spec, values)
	Adds a parameter to the DataSet and associates result values with the new parameter.
	The values must be a NumPy array or a Python list, with each element holding a single result value that matches the parameter's data type.
	If the DataSet is not empty, then the count of provided values must equal the current count of results in the DataSet, or an error will result.
    
    It is an error to add parameters to a completed DataSet.
	
DataSet.mark_complete()
    Marks the DataSet as completed.

Access
------

DataSet.length
    This attribute holds the current number of results in the DataSet. 

DataSet.is_empty
    This attribute will be true if the DataSet is empty (has no results), or false if at least one result has been added to the DataSet.
    It is equivalent to testing if the length is zero.

DataSet.is_marked_complete
    This attribute will be true if the DataSet has been marked as complete or false if it is in progress.

DataSet.get_data(*params, start=, end=)
    Returns the values stored in the DataSet for the specified parameters.
    The values are returned as a list of parallel NumPy arrays, one array per parameter.
    The data type of each array is based on the data type provided when the DataSet was created.
    
    The parameter list may contain a mix of string parameter names, QCoDeS Parameter objects, and ParamSpec objects.
    
    If provided, the start and end parameters select a range of results by result count (index). 
    Start defaults to 0, and end defaults to the current length.
    
    If the range is empty -- that is, if the end is less than or equal to the start, or if start is after the current end of the DataSet –
    then a list of empty arrays is returned.

DataSet.get_parameters()
    Returns a list of ParamSpec objects that describe the parameters stored in this DataSet.

DataSet.get_metadata(tag=)
    Returns metadata for this DataSet.
    
    If a tag string is provided, only metadata stored under that tag is returned.
    Otherwise, all metadata is returned.
    
Subscribing
----------------

DataSet.subscribe(callback, min_wait=, min_count=, state=)
    Subscribes the provided callback function to result additions to the DataSet.
    As results are added to the DataSet, the subscriber is notified by having the callback invoked.
    
    - min_wait is the minimum amount of time between notifications for this subscription, in milliseconds. The default is 100.
    - min_count is the minimum number of results for which a notification should be sent. The default is 1.
    
    When the callback is invoked, it is passed the DataSet itself, the current length of the DataSet, and the state object provided when subscribing.
    If no state object was provided, then the callback gets passed None as the fourth parameter.
    
    The callback is invoked when the DataSet is completed, regardless of the values of min_wait and min_count.
    
    This method returns an opaque subscription identifier.

DataSet.unsubscribe(subid)
    Removes the indicated subscription.
    The subid must be the same object that was returned from a DataSet.subscribe call.

Storage
-------

DataSet.read_from(location, formatter=)
    Reads a DataSet from persistent store.
    Location may be a string file system path, a string URL, or some other string that is meaningful to the formatter specified.
    
    Formatter is a QCoDeS Formatter object that specifies how data is read and written. 
    If not provided, the correct formatter is determined from the file extension and file format.
    If the correct formatter cannot be determined, the default formatter is used. 
    The default formatter is currently GNUPlotFormat().
    
    This is a static method in the DataSet class.
    It returns a new DataSet object.

DataSet.read_updates()
    Updates the DataSet by reading any new results and metadata written since the last read.
    
    This method returns a tuple of two Booleans indicating whether or not there were new results and whether or not there was new metadata.

DataSet.write(location, formatter=, overwrite=)
    Writes the DataSet to persistent store.
    Location may be a string file system path, a string URL, or some other string that is meaningful to the formatter specified.
    
    Formatter is a QCoDeS Formatter object that specifies how data is read and written. 
    If not provided, the default formatter is used; currently the default is GNUPlotFormat().
    
    Overwrite, if true, indicates that any old data found at the specified location should be deleted.
    Otherwise, it is an error to specify a location that is already in use.
    
    This method can be called even if the DataSet is empty, in order to specify the location and format

DataSet.write_updates()
    Writes new results in the DataSet to persistent store.
    Depending on the formatter, this may append to an existing stored version or may overwrite the stored version.

DataSet.write_copy(location, formatter=, overwrite=)
    Writes a separate copy of the DataSet to persistent store.
    Location may be a string file system path, a string URL, or some other string that is meaningful to the formatter specified.
    
    Formatter is a QCoDeS Formatter object that specifies how data is read and written. 
    If not provided, the formatter for the DataSet is used. 
    
    Overwrite, if true, indicates that any old data found at the specified location should be deleted.
    Otherwise, it is an error to specify a location that is already in use.

Open Issues
===========

#. Should DataSets automatically write to persistent store periodically, or should the user be required to call write() in order to flush changes ?

At least for now, it seems useful to maintain the current behavior of the DataSet flushing to disk periodically.
On the other hand, this really isn't core functionality.

**Decision: No, we will leave persistence under control of higher-level code.**

#. Should there be a DataSet method similar to add_result that automatically adds a new result by calling the get() method on all parameters that are defined by QCoDeS Parameters?

It would be really easy to write a helper method that does this, so it doesn’t seem necessary to have it in the core API.

**Decision: No, we will not add such a method.**

#. Should the persistence methods be part of DataSet, or should they be methods on persistence-specific classes?

One advantage of removing them from this class is that it makes DataSet completely stand-alone.
The DataSet module would define two classes, ParamSpec and DataSet, and require only NumPy.
This level of modularity is very desirable.

