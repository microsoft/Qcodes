
|LOGO|
======

.. image:: https://badge.fury.io/py/qcodes.svg
    :target: https://badge.fury.io/py/qcodes

.. image:: https://img.shields.io/badge/source-github-ff69b4.svg
   :target: https://github.com/QCoDeS/Qcodes

.. image:: https://zenodo.org/badge/37137879.svg
   :target: https://zenodo.org/badge/latestdoi/37137879

Qcodes is a Python-based data acquisition framework developed by the Copenhagen / Delft / Sydney / Microsoft quantum computing consortium.

The goal is a common framework for physics experiments, so:

- new students don't need to spend a long time learning software in order to
  participate in experiments
- one has to write their own code only for pieces that are very specific
  to their own experiment
- code can and should be contributed back to the framework
- the process of moving between teams or labs, and of setting up
  a new experiment is streamlined
- physics experiments can take advantage of modern software and best practices


See how easy it is to run for yourself:

.. code:: python

   # imports
   import qcodes as qc
   from qcodes.tests.instrument_mocks import DummyInstrument

   # instantiate two dummy instruments
   # That would be your actual instruments in your application
   mock_dac = DummyInstrument('mock_dac', gates=['ch1', 'ch2'])
   mock_dmm = DummyInstrument('mock_dmm', gates=['v1', 'v2'])

   # optionally (but do it) put the instruments in a station
   station = qc.Station(mock_dac, mock_dmm)

   # define a measurement...
   loop = qc.Loop(mock_dac.ch1.sweep(0, 1, 0.05), 0).each(mock_dmm.v1)

   # ...run it!
   loop.run()

Many more elaborate examples can be found in the example notebooks.



Documentation
-------------

.. toctree::
   :maxdepth: 2

   start/index
   help
   user/index
   community/index
   dataset/index
   api/index
   api/generated/qcodes.instrument_drivers
   roadmap
   changes/index
   examples/index


Indices and tables
==================


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

License
=======

.. include:: ../LICENSE


Contributions
=============

QCoDeS logo contributed by `Guenevere E D K Prawiroatmodjo <bqv662@alumni.ku.dk>`_

.. |LOGO|  image:: ./logos/qcodes_logo.png
   :scale: 50 %
   :alt: qcodes logo
