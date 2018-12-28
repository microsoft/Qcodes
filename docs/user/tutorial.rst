.. _tutorial:

Tutorial
========

.. _driver :

Writing a Driver
----------------

First look at what parameters are and how to create them: `qcodes parameter <https://github.com/QCoDeS/Qcodes/blob/master/docs/examples/Parameters.ipynb>`__ .

Then check out the walk-through to write your first driver:  `qcodes instrument <https://github.com/QCoDeS/Qcodes/blob/master/docs/examples/Creating%20Instrument%20Drivers.ipynb>`__ .

Measuring
---------

Browse the   `example <https://github.com/QCoDeS/Qcodes/blob/master/docs/examples/Measure%20without%20a%20Loop.ipynb>`__ .

.. _simulation :

Simulation
----------

.. todo::  missing


Combined Parameters Sweep
-------------------------

If you want to sweep multiple parameters at once qcodes offers the combine function.
You can combine any number of any kind paramter. 
We'll use a ManualParameter for this example.
One can use both an array, or a set of arrays to pass the setpoint.
We'll cover both cases in this example.


.. code:: python

    import numpy as np

    from qcodes.instrument.parameter import ManualParameter
    from qcodes.utils.validators import Numbers

    gate = ManualParameter('gate', vals=Numbers(-10, 10))
    frequency = ManualParameter('frequency', vals=Numbers(-10, 10))
    amplitude = ManualParameter('amplitude', vals=Numbers(-10, 10))
    # a manual parameter returns a value that has been set
    # so fix it to a value for this example
    amplitude.set(-1)

    combined = qc.combine(gate, frequency, name="gate_frequency")


A name is required, and the combined parameter can now be swept
over, but we have to pass a list of values.

.. code:: python

    # use an array with number_of_parameters x number of setpoints 
    sweep_vals = np.array([[1, 1], [1, 1]])

    loop = qc.Loop(combined.sweep(sweep_vals), delay=0.001).each(amplitude)


.. note:: the set operations are done sequentially, in this case first gate and then frequency is set.
         If the operations are blocking, then the set will block.
         Delay is counted for a step, not for a set. In this case set gate, set frequency and wait 0.0001

This will return this data:

DataSet:
   mode     = DataMode.LOCAL
   location = '2016-10-19/23-05-10'

 +----------+--------------------+----------------+--------------+
 | <Type>   | <array_id>         | <array.name>   | <array.shape>|
 +==========+====================+================+==============+
 | Setpoint | gate_frequency_set | gate_frequency | (2,)         |
 +----------+--------------------+----------------+--------------+
 | Measured | amplitude          | amplitude      | (2,)         |
 +----------+--------------------+----------------+--------------+
 | Measured | gate               | gate           | (2,)         |
 +----------+--------------------+----------------+--------------+
 | Measured | frequency          | frequency      | (2,)         |
 +----------+--------------------+----------------+--------------+

Where  gate_frequency_set contains just the sweep indices and the, perhaps confusingly 
named measured gate and frequency contain the set data.

One can decide to save an aggregated version of the set values, instead of the setpoints 
indices.
To do so one must define an aggregator. In the next example we sweep over x,y,z
and we save the sum of them.

.. code:: python

    x = ManualParameter('x', vals=Numbers(-10, 10))
    y = ManualParameter('y', vals=Numbers(-10, 10))
    z = ManualParameter('z', vals=Numbers(-10, 10))
    p4 = ManualParameter('p4', vals=Numbers(-10, 10))
    # set so we can get a value back
    p4.set(-1)


    def linear(x,y,z):
        return x+y+z

    magnet = qc.combine(x, y, z,
                         name="myvector",
                         units="T",
                         label="magnetic field",
                         aggregator=linear)

    # use number_of_parameters arrays with a length of number of setpoints 
    # note that it will error if the length of the arrays are different
    x_vals = np.linspace(1, 2, 2)
    y_vals = np.linspace(1, 2, 2)
    z_vals = np.linspace(1, 2, 2)

    loop = qc.Loop(magnet.sweep(x_vals, y_vals, z_vals), delay=0.001).each(p4)
    data = loop.run()



   data.myvector_set
   >>> array([ 3.,  6.])


.. _metainstrument :

Meta Instruments
---------------------
The concept of a meta instrument is that of having
two separate Instrument, real or virtual, whose actions can
the be controlled from the meta instrument.
In the following example we will create two dummy instruments and a meta instrument.
All the instruments will live on an InstrumentServer.


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

.. note:: Meta instruments go on a different server from the
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

.. note:: the current architecture is so that you MUST one server per base instrument

The base instrument class stays the same, Meta gets a new method for example:

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

will set both instruments at the same time, say it takes 10 seconds per set,
then setting two things will take 10 seconds, not 20 seconds.

For a complete working example see :download:`this example script <./meta.py>`.

Advanced
--------

.. todo::  missing
