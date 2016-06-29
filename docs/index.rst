Welcome to QCoDeS's documentation!
==================================

Release v\ |version| (:ref:`Installation <install>`)

Qcodes is a Python-based data acquisition framework developed by the Copenhagen / Delft / Sydney / Microsoft quantum computing consortium. 
It's cool!

See how easy it is to measure for yourself:

.. code:: python

    model = AModel()
    gates = MockGates('gates', model=model)
    source = MockSource('source', model=model)
    meter = MockMeter('meter', model=model)
    station = qc.Station(gates, source, meter)
    c0, c1, c2, vsd = gates.chan0, gates.chan1, gates.chan2, source.amplitude
    station.measure()
    data = qc.Loop(c1[-15:15:1], 0.1).loop(c0[-15:12:.5], 0.01).each(
        meter.amplitude, # first measurement, at c2=0 -> amplitude_0 bcs it's action 0
        qc.Task(c2.set, 1), # action 1 -> c2.set(1)
        qc.Wait(0.001),
        meter.amplitude, # second measurement, at c2=1 -> amplitude_4 bcs it's action 4
        qc.Task(c2.set, 0)
        ).run(location='data/test2d', overwrite=True)




Features
--------
- Supports Python  3.3, 3.4 and 3.5
- goal: 100% code coverage with a comprehensive test suite


Documentation
-------------

.. toctree::
   :maxdepth: 2

   user/index
   community/index
   api/index
   roadmap
   changes/index


Indices and tables
==================


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
