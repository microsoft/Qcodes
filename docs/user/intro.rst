.. _introduction:

Introduction
============

.. toctree::
   :maxdepth: 2


Big Picture
-----------

.. todo:: insert big picture

Concepts
--------

The framework is designed for extreme modularity. Don't panic when looking at the source code.
Read the following descriptions of the *pillars* of the framework first.

Instrument
~~~~~~~~~~
.. Description

An instrument is first and most fundamental pillar of qcodes as it represent the hardware you would want to talk to.
One can think of an instrument as the data source be it virtual or real.
The latter case requires a driver to deal with real hardware, whereas the first needs a model that spits data.

An instrument can exist as local instrument or remote instrument, see  :ref:`driver` and/or :ref:`instrument` for more information on the fisrt and :ref:`simulation` for more information on the second.

Due to the architectural limitation of this version, a local instrument does not work with a loop_ .


.. responsibilities
Instruments are responsible of :
  -

.. state
Instruments hold state of:
  -

.. failures
Instruments can fail:
  - 


Parameter
~~~~~~~~~

.. Description
.. responsibilities
.. state
.. failures

Loop
~~~~

.. Description
.. responsibilities
.. state
.. failures

DataSet
~~~~~~~

.. Description
.. responsibilities
.. state
.. failures
