.. _dataset-spec:

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

Metadata
    Many items in this spec have metadata associated with them.
    In all cases, we expect metadata to be represented as a dictionary with string keys.
    While the values are arbitrary and up to the user, in many cases we expect metadata to be nested, string-keyed dictionaries
    with scalars (strings or numbers) as the final values.
    In some cases, we specify particular keys or paths in the metadata that other QCoDeS components may rely on.

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
#. Each DataSet should have a unique identifying string that can be used to create references to DataSets.

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
#. The DataSet identifier should be automatically stored in the DataSet's metadata under the "id" tag.


Writing
----------

#. It should be possible to add a single result or a sequence of results to an in-progress DataSet.
#. It should be possible to add an array of values for a new parameter to an in-progress DataSet.
#. It should be possible to modify a single result or a sequence of results in an in-progress DataSet.
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

#. Storage and persistence should be defined outside of the DataSet class.

The following items are no longer applicable:

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

``ParamSpec(name, type, metadata=)``
    Creates a parameter specification with the given name and type.
    The type should be a NumPy dtype object.

    If metadata is provided, it is included in the overall metadata of the DataSet.
    The metadata can be any JSON-able object.

``ParamSpec.name``
    The name of this parameter.

``ParamSpec.type``
    The dtype of this parameter.

``ParamSpec.metadata``
    The metadata of this parameter.
    This should be an empty dictionary as a default.

Either the QCoDeS Parameter class should inherit from ParamSpec, or the Parameter class should provide
a simple way to get a ParamSpec for the Parameter.

DataSet
~~~~~~~

Construction
------------

``DataSet(name)``
    Creates a DataSet with no parameters.
    The name should be a short string that will be part of the DataSet's identifier.

``DataSet(name, specs)``
    Creates a DataSet for the provided list of parameter specifications.
    The name should be a short string that will be part of the DataSet's identifier.
    Each item in the list should be a ParamSpec object.

``DataSet(name, specs, values)``
    Creates a DataSet for the provided list of parameter specifications and values.
    The name should be a short string that will be part of the DataSet's identifier.
    Each item in the specs list should be a ParamSpec object.
    Each item in the values list should be a NumPy array or a Python list of values for the corresponding ParamSpec.
    There should be exactly one item in the values list for every item in the specs list.
    All of the arrays/lists in the values list should have the same length.
    The values list may intermix NumPy arrays and Python lists.

``DataSet.add_parameter(spec)``
    Adds a parameter to the DataSet.
    The spec should be a ParamSpec object.
    If the DataSet is not empty, then existing results will have the type-appropriate null value for the new parameter.

    It is an error to add parameters to a completed DataSet.

``DataSet.add_parameters(specs)``
    Adds a list of parameters to the DataSet.
    Each item in the list should be a ParamSpec object.
    If the DataSet is not empty, then existing results will have the type-appropriate null value for the new parameters.

    It is an error to add parameters to a completed DataSet.

``DataSet.add_metadata(tag=, metadata=)``
    Adds metadata to the DataSet.
    The metadata is stored under the provided tag.
    If there is already metadata under the provided tag, the new metadata replaces the old metadata.
    The metadata can be any JSON-able object.

Writing
-------

``DataSet.add_result(**kwargs)``
    Adds a result to the DataSet.
    Keyword parameters should have the name of a parameter as the keyword and the value to associate as the value.
    If there is only one positional parameter and it is a dictionary, then it is interpreted as a map from parameter name to parameter value.

    Returns the zero-based index in the DataSet that the result was stored at; that is, it returns the length of the DataSet before the addition.

    It is an error to provide a value for a key or keyword that is not the name of a parameter in this DataSet.

    It is an error to add a result to a completed DataSet.

``DataSet.add_results(args)``
    Adds a sequence of results to the DataSet.
    The single argument should be a sequence of dictionaries, where each dictionary provides the values for all of the parameters in that result.
    See the add_result method for a description of such a dictionary.
    The order of dictionaries in the sequence will be the same as the order in which they are added to the DataSet.

    Returns the zero-based index in the DataSet that the first result was stored at; that is, it returns the length of the DataSet before the addition.

    It is an error to provide a value for a key or keyword that is not the name of a parameter in this DataSet.

    It is an error to add results to a completed DataSet.

``DataSet.modify_result(index, **kwargs)``
    Modifies a result in the DataSet.
    The index should be the zero-based index of the result to be modified.
    Keyword parameters should have the name of a parameter as the keyword and the updated value to associate as the value.
    If there is only one positional parameter and it is a dictionary, then it is interpreted as a map from parameter name to updated parameter value.

    Any parameters that were specified in the original result that do not appear in the modification are left unchanged.
    To remove a parameter from a result, map it to None.

    It is an error to modify a result at an index less than zero or beyond the end of the DataSet.

    It is an error to provide a value for a key or keyword that is not the name of a parameter in this DataSet.

    It is an error to modify a result in a completed DataSet.

``DataSet.modify_results(start_index, updates)``
    Modifies a sequence of results in the DataSet.
    The start_index should be the zero-based index of the first result of the sequence to be modified.
    The updates argument should be a sequence of dictionaries, where each dictionary provides modified values for parameters
    as a map from parameter name to parameter value.
    See the modify_result method for a description of such a dictionary.
    The order of dictionaries in the sequence will be the same as the order in which they are applied to the DataSet.

    Any parameters that were specified in a original result that do not appear in the corresponding modification are left unchanged.
    To remove a parameter from a result, map it to None.

    It is an error to modify a result at an index less than zero or beyond the end of the DataSet.

    It is an error to provide a value for a key or keyword that is not the name of a parameter in this DataSet.

    It is an error to modify results in a completed DataSet.

``DataSet.add_parameter_values(spec, values)``
    Adds a parameter to the DataSet and associates result values with the new parameter.
    The values must be a NumPy array or a Python list, with each element holding a single result value that matches the parameter's data type.
    If the DataSet is not empty, then the count of provided values must equal the current count of results in the DataSet, or an error will result.

    It is an error to add parameters to a completed DataSet.

``DataSet.mark_complete()``
    Marks the DataSet as completed.

Access
------

``DataSet.id``
    Returns the unique identifying string for this DataSet.
    This string will include the date and time that the DataSet was created and the name supplied to the constructor,
    as well as additional content to ensure uniqueness.

``DataSet.length``
    This attribute holds the current number of results in the DataSet.

``DataSet.is_empty``
    This attribute will be true if the DataSet is empty (has no results), or false if at least one result has been added to the DataSet.
    It is equivalent to testing if the length is zero.

``DataSet.is_marked_complete``
    This attribute will be true if the DataSet has been marked as complete or false if it is in progress.

``DataSet.get_data(*params, start=, end=)``
    Returns the values stored in the DataSet for the specified parameters.
    The values are returned as a list of parallel NumPy arrays, one array per parameter.
    The data type of each array is based on the data type provided when the DataSet was created.

    The parameter list may contain a mix of string parameter names, QCoDeS Parameter objects, and ParamSpec objects.

    If provided, the start and end parameters select a range of results by result count (index).
    Start defaults to 0, and end defaults to the current length.

    If the range is empty -- that is, if the end is less than or equal to the start, or if start is after the current end of the DataSet –
    then a list of empty arrays is returned.

``DataSet.get_parameters()``
    Returns a list of ParamSpec objects that describe the parameters stored in this DataSet.

``DataSet.get_metadata(tag=)``
    Returns metadata for this DataSet.

    If a tag string is provided, only metadata stored under that tag is returned.
    Otherwise, all metadata is returned.

Subscribing
----------------

``DataSet.subscribe(callback, min_wait=, min_count=, state=)``
    Subscribes the provided callback function to result additions to the DataSet.
    As results are added to the DataSet, the subscriber is notified by having the callback invoked.

    - min_wait is the minimum amount of time between notifications for this subscription, in milliseconds. The default is 100.
    - min_count is the minimum number of results for which a notification should be sent. The default is 1.

    When the callback is invoked, it is passed the DataSet itself, the current length of the DataSet, and the state object provided when subscribing.
    If no state object was provided, then the callback gets passed None as the fourth parameter.

    The callback is invoked when the DataSet is completed, regardless of the values of min_wait and min_count.

    This method returns an opaque subscription identifier.

``DataSet.unsubscribe(subid)``
    Removes the indicated subscription.
    The subid must be the same object that was returned from a DataSet.subscribe call.

Storage
-------

DataSet persistence is handled externally to this class.

The existing QCoDeS storage subsystem should be modified so that some object has two methods:

- A write_dataset method that takes a DataSet object and writes it to the appropriate storage location in an appropriate format.
- A read_dataset method that reads from the appropriate location, either with a specified format or inferring the format, and returns
  a DataSet object.

Metadata
========

While in general the metadata associated with a DataSet is free-form, it is useful to specify a set of "well-known" tags and paths that components can rely on to contain specific information.
Other components are free to specify new well-known metadata tags and paths, as long as they don't conflict with the set defined here.

parameters
    This tag contains a dictionary from the string name of each parameter to information about that parameter.
    Thus, if DataSet ds has a parameter named "foo", there will be a key "foo" in the dictionary returned from ds.get_metadata("parameters").
    The value associated with this key will be a string-keyed dictionary.

parameters/__param__/spec
    This path contains a string-keyed dictionary with (at least) the following two keys:
    The "type" key is associated with the NumPy dtype for the values of this parameter.
    The "metadata" key is associated with the metadata that was passed to the ParamSpec constructor that defines this parameter, or an empty dictionary if no metadata was set.

Utilities
=========

There are many utility routines that may be defined outside of the DataSet class that may be useful.
We collect several of them here, with the note that these functions will not be part of the DataSet class
and will not be required by the DataSet class.

dataframe_from_dataset(dataset)
    Creates a Pandas DataFrame object from a DataSet that has been marked as completed.

Open Issues
===========

#. Should it be possible to "reopen" a DataSet that has been marked as completed?

This is convenient for adding data analysis results after the experiement has added, but could potentially lead mixing data from different experimental runs accidentally.
It is already possible to modify metadata after the DataSet has beenmarked as completed, but sometimes that may not be sufficient.
