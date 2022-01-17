Object Hierarchy
================

.. todo:: make sure it is updated and easy to read.

Rough linkages:
---------------

In **bold** the containing class creates this object. In *italics* the
container just holds this object (or class) as a default for derivatives
to use. Normal text shows the container includes and uses of this object.

-  Station
-  Instrument: IPInstrument, VisaInstrument, MockInstrument

   -  **Parameter**
   -  Validator: Anything, Strings, Numbers, Ints, Enum, MultiType
   -  **SweepValues**: SweepFixedValues, AdaptiveSweep
   -  Function
   -  Validator

-  **Monitor**
-  *actions*
-  DataManager
-  **DataServer**
-  :ref:`loops_api`
-  actions: Parameter, Task, Wait, (Active)Loop
-  **ActiveLoop**

   -  **DataSet**
   -  **DataArray**
   -  **Formatter**: GNUPlotFormat
   -  **DiskIO** (may become subclass of IOManager?)
   -  **FormatLocation** (a location\_provider)

Station
-------

A convenient container for instruments, parameters, and more.

More information:

- `Station example notebook <../examples/Station.ipynb>`_
- :ref:`station_api` API reference


.. _instrument :

Instrument
----------

A representation of one particular piece of hardware.

Lists all ``Parameter``\ s and ``Function``\ s this hardware is
capable of, and handles the underlying communication with the
hardware.  ``Instrument`` sets the common structure but real
instruments will generally derive from its subclasses, such as
``IPInstrument`` and ``VisaInstrument``. There is also
``MockInstrument`` for making simulated instruments, connected to a
``Model`` that mimics a serialized communication channel with an
apparatus.


.. todo:: add all the things from the  github issue

Parameter
---------

A representation of one particular state variable.

Most ``Parameter``\ s are part of an ``Instrument``, and parameter is
linked to specific commands that are sent to a specific instrument.
But you can also create ``Parameter``\ s that
execute arbitrary functions, for example to combine several gate
voltages in a diagonal sweep. Parameters can have setters and/or getters
(they must define at least a setter OR a getter but do not need to
define both)

Gettable ``Parameter``\ s often return one value, but can return a
collection of values - several values with different names, or one or
more arrays of values, ie if an instrument provides a whole data set at
once, or if you want to save raw and calculated data side-by-side

Settable ``Parameter``\ s take a single value, and can be sliced to
create a ``SweepFixedValues`` object.

Validator
---------

Defines a set of valid inputs, and tests values against this set

Subclasses include ``Anything``, ``Strings``, ``Numbers``, ``Ints``,
``Enum``, and ``MultiType``, each of which takes various arguments that
restrict it further.

SweepValues
-----------

An iterator that provides values to a sweep, along with a setter for
those values connected to a ``Parameter``.

Mostly the ``SweepFixedValues`` subclass is used (this is flexible
enough to iterate over any arbitrary collection of values, not
necessarily linearly spaced) but ot
