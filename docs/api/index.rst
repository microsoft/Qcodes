.. currentmodule:: qcodes

Classes and Functions
=====================

This page lists the entrypoints to the qcodes API.

.. note::
   this should be in sync with the qcodes namespace

.. toctree::
   :maxdepth: 4

Station
~~~~~~~

.. autosummary::
   :toctree: generated/

   station.Station

Loops
~~~~~

.. autosummary::
   :toctree: generated/

   get_bg
   halt_bg
   Loop


Actions
~~~~~~~

.. autosummary::
   :toctree: generated/

   Task
   Wait
   BreakIf

Data
~~~~

.. autosummary::
   :toctree: generated/

    get_data_manager
    DataMode
    DataSet
    new_data
    load_data
    FormatLocation
    DataArray
    Formatter
    GNUPlotFormat
    DiskIO



Instrument
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Function
   Parameter
   StandardParameter
   SweepValues

   Instrument
   IPInstrument
   VisaInstrument
   MockInstrument
   MockModel

Utils
~~~~~
.. automodule:: 

.. autosummary::
   :toctree: generated/

   qcodes.utils.validators
