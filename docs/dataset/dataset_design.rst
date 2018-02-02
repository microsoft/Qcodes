.. highlight:: python

==============
Dataset Design
==============

.. _sec:design_introduction:

Introduction
============

This document aims to explain the design and working of the QCoDeS DataSet.
In :numref:`datasetdiagram` we sketch the basic design of the dataset.
The dataset implementation is organised in 3 layers shown vertically in
:numref:`datasetdiagram` The lower most layer implements the direct
communication with the database. It's expected that the user will not need
to interact with this layer directly


.. _datasetdiagram:
.. figure:: figures/datasetdiagram.svg
   :align: center
   :width: 100%

   Basic workflow




`measurement.py` implements a context manager that enables




Possible Future Directions
==========================


