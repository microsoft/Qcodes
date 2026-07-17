.. highlight:: python

============
Introduction
============

This page explains the basics of the QCoDeS dataset. Read here to figure out what the heck is going on.

.. _sec:intro_basics:

Basics
======

The DataSet is directly linked to an SQLite database on the host machine's hard drive. Here we provide an overview of the layout of that database.
The specifics of the python objects implementing the API on top of that can be found in the section :ref:`datasetdesign`.

The database holds a number of ``Experiments`` that each holds a number of ``DataSets``. One ``DataSet`` contains exactly one ``run``, and we may use the words ``DataSet`` and ``run`` interchangeably.

.. _sec:intro_experiment:

Experiment
==========

The intended use of the ``Experiment`` is to group ``DataSets`` that belong together. For example, an experiment may contain all the measurements necessary to characterise a particular sample (Is it superconducting? What is the critical current?). Once the goodness of the sample has been established, a new experiment containing all the "real" measurements can then commence. The grouping runs this way is offered as a convenience and is as such not essential for any functionality in QCoDeS.

Each ``Experiment`` has the following attributes:

 * An experiment-ID
 * A name
 * A sample-name
 * A start time
 * (potentially) An end time

The **experiment-ID** is an ever-increasing integer enumerating the number of experiments in the current database. The experiments are kept in the SQLite database in the ``experiments`` table. The experiment-ID is the primary key of this table.

The **name** and the **sample name** of an experiment must be provided by the user at creation time.

The **start time** is automatically added upon creation, and the **end time** is added when (if ever) the user decides to mark the experiment as *completed* after which point no more runs may be added.

For an example notebook showing the usage of the database as a container for experiments, see the :doc:`Experiment Container Notebook <../examples/DataSet/The-Experiment-Container>`.


.. _sec:intro_dataset:

DataSet
=======

Each ``DataSet`` has the following attributes:

  * A run-ID
  * An experiment-ID
  * A name
  * A table of results
  * A number of parameters
  * A start time
  * (potentially) An end time
  * A GUID

The **run-ID** is an ever-increasing integer enumerating the number of runs in the current database. The runs are kept in the SQLite database in the ``runs`` table. The run-ID is the primary key of this table.

The **experiment-ID** provides the link to the experiment that this dataset belongs to.

The **name** must be provided by the user at creation time.

The **table of results** is where the actual *data* of the ``DataSet`` resides. The table has a number of associated **parameters** as columns with data points as rows. The parameters may be QCoDeS ``Parameters``, but are not limited to that. The preferred procedure for associating parameters with a run and adding data is via the ``Measurement`` object as described in the :doc:`Measurement Object Example Notebook <../examples/DataSet/Performing-measurements-using-qcodes-parameters-and-dataset>`. The deep thinking behind how different parameters relate to and depend on each other within a ``DataSet`` is explained in :ref:`interdependentparams`.

The **start time** is automatically added upon creation, and the **end time** is added when (if ever) the user decides to mark the run as *completed* after which point no more data points may be added.

The **GUID** is a run identifier that aims to go `beyond` the current database. The motivation for the GUID is that collaboration and data sharing across several machines requires a run to be labelled by something more than its name and number in the local database. The GUID is a standard 36 character string GUID composed from three integer codes and a timestamp. The codes are: location code (1-256), work station code (1-16777216), sample code (1-4294967296). The idea is that the people in a certain collaboration will label their locations, machines, and samples according to a scheme agreed upon `within` the relevant collaboration. That way, each run is uniquely identifiable in a human-readable way. This extra label for runs is offered as a convenience as is as such not essential for any functionality in QCoDeS.

For an example notebook showing details of ``DataSet`` class and its usage, see the :doc:`DataSet class walkthrough notebook <../examples/DataSet/DataSet-class-walkthrough>` and :doc:`Accessing data in DataSet notebook <../examples/DataSet/Accessing-data-in-DataSet>`.

For dataset operations, QCoDeS provides functions for:

- **Exporting datasets**: :doc:`Exporting data to other file formats <../examples/DataSet/Exporting-data-to-other-file-formats>`
- **Extracting runs between databases**: :doc:`Extracting runs from one DB file to another <../examples/DataSet/Extracting-runs-from-one-DB-file-to-another>` and :func:`qcodes.dataset.extract_runs_into_db`
- **Bulk export and metadata-only databases**: :func:`qcodes.dataset.export_datasets_and_create_metadata_db` for creating lightweight metadata-only databases while exporting all data to NetCDF files

.. _sec:intro_split_raw_data:

Split Raw Data Storage
======================

By default, all measurement data (the results table rows) is stored in the same SQLite database alongside metadata such as experiments, runs, parameter layouts, and dependencies. Over time, the main database file can grow very large, which can slow down operations like browsing experiments and loading metadata.

QCoDeS supports an optional **split raw data storage** mode in which the actual measurement data for each ``DataSet`` is written to an individual, per-dataset SQLite file while all metadata remains in the main database. Each per-dataset file is named after the dataset's GUID (e.g. ``<guid>.db``) and is stored in a configurable folder.

This feature is controlled by two configuration options in ``qcodesrc.json``:

- ``dataset.raw_data_to_separate_db`` (bool, default ``false``): enables or disables split storage.
- ``dataset.raw_data_path`` (string, default ``"{db_location}"``): the folder where per-dataset files are created. The ``{db_location}`` placeholder is expanded to a folder derived from the main database path (e.g. ``~/experiments.db`` becomes ``~/experiments_db/``).

When enabled:

- The main database retains the full results table schema (column definitions) but no data rows are written to it, keeping it lightweight.
- All ``INSERT`` and ``SELECT`` operations on results data are transparently routed to the per-dataset file.
- The path to the per-dataset file is persisted in the run's metadata (``raw_data_db_path``), so ``load_by_id`` and related loading functions automatically reconnect to the correct file.
- All public ``DataSet`` APIs (``get_parameter_data``, ``to_pandas_dataframe``, ``to_xarray_dataset``, ``cache``, ``export``, etc.) work identically whether split storage is enabled or not.

Example runtime configuration::

    import qcodes as qc

    qc.config.dataset.raw_data_to_separate_db = True
    qc.config.dataset.raw_data_path = "/data/raw_measurements/"

If the per-dataset raw data files are moved to a different folder (e.g. during data migration or archival), the stored paths in the main database will become stale. Use the :func:`~qcodes.dataset.update_raw_data_paths` helper to update them::

    from qcodes.dataset import update_raw_data_paths

    update_raw_data_paths(
        db_path="/path/to/main_database.db",
        new_raw_data_folder="/new/location/of/raw_files/"
    )

This scans all datasets with a ``raw_data_db_path`` metadata entry, checks whether the corresponding ``.db`` file exists in the new folder, and updates the stored path accordingly.

Managing Datasets
-----------------

QCoDeS provides helpers for managing per-dataset raw data files over time:

- :func:`~qcodes.dataset.purge_orphaned_datasets` — remove dataset records whose raw data files no longer exist on disk.
- :func:`~qcodes.dataset.cleanup_datasets` — remove datasets (DB records and raw data files) by age, sample name, or file size.

For usage examples, see the :doc:`Database notebook <../examples/DataSet/Database>`.
