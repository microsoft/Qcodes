.. _public :

.. currentmodule:: qcodes

Public
======

Classes and Functions
----------------------

This page lists the entrypoints to the plubic qcodes API.

.. toctree::
   :maxdepth: 4


.. _config_label:

Config
~~~~~~

.. autosummary::
   :toctree: generated/

   Config

.. _station:

Station
~~~~~~~

.. autosummary::
   :toctree: generated/

   station.Station

.. _loops:

Loops
~~~~~

.. autosummary::
   :toctree: generated/

   Loop

Measure
~~~~~~~

.. autosummary::
   :toctree: generated/

   measure.Measure

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
   combine
   CombinedParameter


   Instrument
   IPInstrument
   VisaInstrument


Plot
~~~~
Note that the plotting modules may not be available if their dependencies were not met during installation of the package.

.. autosummary::
   :toctree: generated/

    qcodes.plots.qcmatplotlib.MatPlot
    qcodes.plots.pyqtgraph.QtPlot

Utils & misc
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   qcodes.utils.validators
   qcodes.utils.helpers.in_notebook

