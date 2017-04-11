
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
- code can and should contributed back to the framework
- the process of moving between teams or labs, and of setting up
  a new experiment is streamlined
- physics experiments can take advantage of modern software and best practices 


See how easy it is to run for yourself:

.. code:: python

    # define your systesm
    model = AModel()
    # define your instruments
    gates = MockGates('gates', model=model)
    source = MockSource('source', model=model)
    meter = MockMeter('meter', model=model)
    # optionally group all the components in a station
    station = qc.Station(gates, source, meter)
    c0, c1, c2, vsd = gates.chan0, gates.chan1, gates.chan2, source.amplitude
    # and loop
    data = qc.Loop(c1[-15:15:1], 0.1).loop(c0[-15:12:.5], 0.01).each(
        # perform actiosn at heak
        meter.amplitude, # first measurement, at c2=0 -> amplitude_0 bcs it's action 0
        qc.Task(c2.set, 1), # action 1 -> c2.set(1)
        qc.Wait(0.001),
        meter.amplitude, # second measurement, at c2=1 -> amplitude_4 bcs it's action 4
        qc.Task(c2.set, 0)
        # finally save
        ).run(location='data/test2d', overwrite=True)



Documentation
-------------

.. toctree::
   :maxdepth: 2

   start/index 
   help
   user/index
   community/index
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
