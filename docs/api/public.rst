.. _public :

.. currentmodule:: qcodes

Public
======

Classes and Functions
----------------------

This page lists the entrypoints to the plubic qcodes API.

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
   SweepFixedValues
   SweepValues

   Instrument
   IPInstrument
   VisaInstrument
   MockInstrument
   MockModel


Plot
~~~~
Note that the plotting modules may not be available if their dependencies were not met during installation of the package.

.. autosummary::
   :toctree: generated/

    qcodes.plots.qcmatplotlib.MatPlot
    qcodes.plots.qcmatplotlib.QtPlot

Utils & misc
~~~~~~~~~~~~
.. automodule::

.. autosummary::
   :toctree: generated/

   qcodes.utils.validators
   qcodes.process.helpers.set_mp_method
   qcodes.utils.helpers.in_notebook
   qcodes.widgets.widgets.show_subprocess_widget

