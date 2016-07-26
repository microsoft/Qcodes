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


Composite Instruments
--------------------- 
The concept of a composite instrument is that of having
two separate Instrument, real or virtual, whose actions can
the be controlled from the composite instrument.
In the following example we will create two dummy instruments and a composite instruments.
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

Then we create the composite instrument, this will hold any of the base 
instruments.
Since we want the composite instrument to be able to talk to the base instrumetns 
we need to include a list of them as shared_kwargs.


.. note:: Every InstrumentServer needs to have identical shared_kwargs among all the instruments loaded there. That's because these args get loaded into the server when it's created, then passed on to each instrument that's loaded there during its construction on the server side.  

.. code:: python

    class base1base2(Instrument):
        shared_kwargs = ['instruments']

        # Instruments will be a list of RemoteInstrument objects, which can be
        # given to a server on creation but not later on, so it needs to be
        # listed in shared_kwargs
        def __init__(self, name, instruments=(), **kwargs):
            super().__init__(name, **kwargs)
            self._instrument_list = instruments

            for gate in range(len(self._instrument_list)):
                self.add_parameter('c%d' % gate,
                                   get_cmd=partial(self._get, gate=gate),
                                   set_cmd=partial(self._set, gate=gate))

            self.add_parameter("setBoth", set_cmd= partial(self._set_both))

        def _set_both(self, value):
            for i in self._instrument_list:
                i.set('x', value)

        def _get(self, gate):
            logging.debug('base1base2._get: %s' % (gate,))
            return self._instrument_list[gate].get('x')

        def _set(self, value, gate):
            logging.debug('base1base2._set: gate %s, value %s' % (gate, value))
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
    composite_server_name
    composite = base1base2(name='composite', server_name=composite_server_name,
                      instruments=[base1, base2])

.. notes:: Composite instruments go on a different server from the
    low-level instruments it references, because reaons.


And now one case use the composite as expected:

.. code:: python

    print("--- set composite --- ")
    composite.c1.set(25)
    print(composite.c1.get())
    >>> 25
    print(base1.x.get())
    >>> 25

    print("--- set base --- ")
    base1.x.set(1)
    print(composite.c1.get())
    >>> 1
    print(base1.x.get())
    >>> 1

    
    composite.setBoth(0)
    print(base1.x.get())
    >>> 0
    print(base0.x.get())
    >>> 0


Async Composite
~~~~~~~~~~~~~~~
Say you want to set two instruments at the same time. 
You can use the following:

.. note:: the curernt architecture is so b0rken, you MUST one  server per base instrument

The base instrument classe stays the same, composite gets a new method f.ex:

.. code:: python

    class base1base2(Instrument):
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
            self.add_parameter("setBoth", set_cmd= partial(self._set_both))
            self.add_parameter("setBothAsync", set_cmd= partial(self._set_async))

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
            logging.debug('base1base2._get: %s' % (gate,))
            return self._instrument_list[gate].get('x')

        def _set(self, value, gate):
            logging.debug('base1base2._set: gate %s, value %s' % (gate, value))
            i = self._instrument_list[gate]
            i.set('x', value)

        # note the different server names
        # that's required
        base0 = MyInstrument(name='zero', server_name="foo")
        base1 = MyInstrument(name='one', server_name="bar")

        composite_server_name = "composite_server"
        composite = base1base2(name='composite', server_name=composite_server_name,
                          instruments=[base0, base1])

This way:  
    >>> composite.setBothAsync(0)

will set both instrument at the same time, say it takes 10 seconds per set, 
then setting two things will take 10 seconds, not 20 seconds.



Avanced
-------

.. todo::  missing
