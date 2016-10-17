.. _tutorial:

Tutorial
========

In this tutorial we'll walk through *****


.. _driver :

Writing a Driver
----------------

Write a simple driver example
with commented code
  - add parameter
  - add validator
  - add custom stuff
  - add doccstrings f.ex

.. todo::  missing


Measuring
---------

.. todo::  missing

.. _simulation :

Simulation
----------
Explain the mock mock

.. todo::  missing


.. __metainstrument :

Meta Instruments
---------------------
The concept of a meta instrument is that of having
two separate Instrument, real or virtual, whose actions can
the be controlled from the meta instrument.
In the following example we will create two dummy instruments and a meta instruments.
All the instruments will live on a InstrumentServer.


.. note:: this is rather non-trival due to the limitation of the
    current multiprocessing architecture

First we create an instrument:

.. code:: python

    class MyInstrument(Instrument):

        def __init__(self, name, **kwargs):
            super().__init__(name, **kwargs)
            self.x=0
            self.add_parameter('x',  get_cmd=self.getx, set_cmd=self.setx)

        def getx(self):
            return self.x

        def setx(self, val):
            self.x=val

Then we create the meta instrument, this will hold any of the base
instruments.
Since we want the meta instrument to be able to talk to the base instruments
we need to include a list of them as shared_kwargs.


.. note:: Every InstrumentServer needs to have identical shared_kwargs among all the instruments loaded there. That's because these args get loaded into the server when it's created, then passed on to each instrument that's loaded there during its construction on the server side.

.. code:: python

    class Meta(Instrument):
        shared_kwargs = ['instruments']

        # Instruments will be a list of RemoteInstrument objects, which can be
        # given to a server on creation but not later on, so it needs to be
        # listed in shared_kwargs

        def __init__(self, name, instruments=(), **kwargs):
            super().__init__(name, **kwargs)
            self._instrument_list = instruments
            self.no_instruments = len(instruments)
            for gate in range(len(self._instrument_list)):
                self.add_parameter('c%d' % gate,
                                   get_cmd=partial(self._get, gate=gate),
                                   set_cmd=partial(self._set, gate=gate))

            self.add_parameter("setBoth", set_cmd=partial(self._set_both))
            self.add_parameter("setBothAsync", set_cmd=partial(self._set_async))

        def _set_both(self, value):
            for i in self._instrument_list:
                i.set('x', value)

        def _get(self, gate):
            value =self._instrument_list[gate].get('x')
            logging.debug('Meta get gate %s' % (value))
            return value

        def _set(self, value, gate):
            logging.debug('Meta set gate %s @ value %s' % (gate, value))
            i = self._instrument_list[gate]
            i.set('x', value)

Let's put these babies on servers:

.. code:: python

   BASESERVER = "foo"
   base1 = VirtualIVVI(name='one', server_name=BASESERVER)
   base2 = VirtualIVVI(name='two', server_name=BASESERVER)

.. note:: Instruments with no shared_kwargs  can go on the same or different servers.
          That means that base1 and base2 don't know about eachoter.

.. code:: python

    meta_server_name = "meta_server"
    meta = Meta(name='meta', server_name=meta_server_name,
                      instruments=[base1, base2])

.. notes:: Meta instruments go on a different server from the
    low-level instruments it references, because reasons.


And now one case use the meta as expected:

.. code:: python

    print("--- set meta --- ")
    meta.c1.set(25)
    print(meta.c1.get())
    >>> 25
    print(base1.x.get())
    >>> 25

    print("--- set base --- ")
    base1.x.set(1)
    print(meta.c1.get())
    >>> 1
    print(base1.x.get())
    >>> 1

    meta.setBoth(0)
    print(base1.x.get())
    >>> 0
    print(base0.x.get())
    >>> 0



Async Meta
----------

Say you want to set two instruments at the same time.
You can use the following:

.. note:: the curernt architecture is so that you MUST one  server per base instrument

The base instrument class stays the same, meta gets a new method f.ex:

.. code:: python

    class Meta(Instrument):
        shared_kwargs = ['instruments']

        # Instruments will be a list of RemoteInstrument objects, which can be
        # given to a server on creation but not later on, so it needs to be
        # listed in shared_kwargs
        def __init__(self, name, instruments=(), **kwargs):
            super().__init__(name, **kwargs)
            self._instrument_list = instruments
            self.no_instruments = len(instruments)
            for gate in range(len(self._instrument_list)):
                self.add_parameter('c%d' % gate,
                                   get_cmd=partial(self._get, gate=gate),
                                   set_cmd=partial(self._set, gate=gate))
            self.add_parameter("setBoth", set_cmd=partial(self._set_both))
            self.add_parameter("setBothAsync", set_cmd=partial(self._set_async))

        def _set_both(self, value):
            for i in self._instrument_list:
                i.set('x', value)

        def _set_async(self, value):
            with futures.ThreadPoolExecutor(max_workers=self.no_instruments) as executor:
                jobs = []
                for i in self._instrument_list:
                    job = executor.submit(partial(i.set, 'x'), value)
                    jobs.append(job)
                futures.wait(jobs)

        def _get(self, gate):
            value =self._instrument_list[gate].get('x')
            logging.debug('Meta get gate %s' % (value))
            return value

        def _set(self, value, gate):
            logging.debug('Meta set gate %s @ value %s' % (gate, value))
            i = self._instrument_list[gate]
            i.set('x', value)


This way:
    >>> meta.setBothAsync(0)

will set both instrument at the same time, say it takes 10 seconds per set,
then setting two things will take 10 seconds, not 20 seconds.

For a complete working example see :download:`this example script <./meta.py>`.

Avanced
-------

.. todo::  missing
