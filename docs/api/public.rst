.. _public :

.. currentmodule:: qcodes

Public
======

Classes and Functions
---------------------

This page lists the entrypoints to the public qcodes API.

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

DataSet
~~~~~~~

.. autosummary::
   :toctree: generated/

    qcodes.dataset.measurements.Measurement
    qcodes.dataset.measurements.DataSaver

    qcodes.dataset.experiment_container.Experiment
    qcodes.dataset.experiment_container.new_experiment
    qcodes.dataset.experiment_container.load_last_experiment
    qcodes.dataset.experiment_container.load_experiment_by_name
    qcodes.dataset.experiment_container.load_or_create_experiment

    qcodes.dataset.database.initialise_database
    qcodes.dataset.database.initialise_or_create_database_at

    qcodes.dataset.data_set.load_by_id

    qcodes.dataset.plotting.plot_by_id

Instrument
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Function
   Parameter
   StandardParameter
   ArrayParameter
   MultiParameter
   ManualParameter
   SweepFixedValues
   SweepValues
   combine
   CombinedParameter


   Instrument
   IPInstrument
   VisaInstrument

   InstrumentChannel
   ChannelList

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
   qcodes.utils.plotting
   qcodes.logger
