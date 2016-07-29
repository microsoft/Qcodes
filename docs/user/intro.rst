.. _introduction:

Introduction
============

.. toctree::
   :maxdepth: 2


Big Picture
-----------

As computers and electronic hardware have gotten more powerful and pervasive, several quantities involved in computer-controlled experiments (and simulations) continue to increase:

  - the number of instruments involved in any experiment
  - the frequency with which users move those instruments around
  - the number of settings available on each instrument
  - the amount and variety of data produced

QCoDeS is designed to manage that complexity, giving a consistent way to control and loop over all the settings of all the instruments, tracking what instruments are connected and what their settings were at any time during the experiment, structuring the data for clear analysis, and sharing code to minimize the setup time on a new experiment.

Concepts
--------

The framework is designed for extreme modularity. Don't panic when looking at the source code.
Read the following descriptions of the *pillars* of the framework first.

Instrument
~~~~~~~~~~
.. Description

An instrument is first and most fundamental pillar of qcodes as it represent the hardware you would want to talk to, either to control your system, collect data, or both. Instruments come in several flavors:

  - Hardware: most instruments map one-to-one to a real piece of hardware; in these instances, the QCoDeS Instrument requires a driver or communication channel to the hardware. See  :ref:`driver` and/or :ref:`instrument`.

  - Simulation: for theoretical or computational work, an instrument may contain or connect to a model that has its own state and generates results in much the same way that an instrument returns measurements. See :ref:`simulation`.

  - Manual: If a real instrument has no computer interface (just physical switches and knobs), you may want a QCoDeS instrument for it just to maintain a record of how it was configured at any given time during the experiment, and perhaps to use those settings in calculations. In these cases it is of course up to the user to keep the hardware and software synchronized.

  - Meta: Sometimes to make the experiment easier to manage, it is useful to make an instrument to represent some element of the system that may be controlled in part by several separate instruments. For example, a device that uses one instrument to supply DC voltages, another to supply AC signals, and a third to measure it. We can make a new QCoDeS Instrument that references these lower-level Instruments, and we refer to this as a "Meta-Instrument". This imposes some requirements and limitations with remote instruments and multiple processes - see :ref:`metainstrument` for more information.

An instrument can exist as local instrument or remote instrument. A local instrument is instantiated in the main process, and you interact with it directly. This is convenient for testing and debugging, but a local instrument cannot be used with a measurement loop_ in a separate process (ie a background loop). For that purpose you need a remote instrument. A remote instrument starts (or connects to) a server process, instantiates the instrument in that process, and returns a proxy object that mimicks the API of the instrument. This proxy holds no state or hardware connection, so it can be freely copied to other processes, in particular background loops and meta-instrument servers.

.. responsibilities
Instruments are responsible for:
  - Holding connections to hardware, be it VISA, some other communication protocol, or a specific DLL or lower-level driver.
  - Creating a parameter_, ref:`function`, or method for each piece of functionality we support. These objects may be used independent of the instrument, but they make use of the instrument's hardware connection in order to do their jobs.
  - Describing their complete current state ("snapshot") when asked, as a JSON-compatible dictionary.

.. state
Instruments hold state of:
  - The communication address, and in many cases the open communication channel.
  - The most recent set or measured value of every parameter_, so that it does not need to query the hardware for every value when asked for a snapshot.

.. failures
Instruments can fail:
  - When a VisaInstrument has been instantiated before, particularly with TCPIP, sometimes it will complain "VI_ERROR_RSRC_NFOUND: Insufficient location information or the requested device or resource is not present in the system" and not allow you to open the instrument until either the hardware has been power cycled or the network cable disconnected and reconnected. Are we using visa/pyvisa in a brittle way?
  - If you try to use a background loop_ with a local instrument, because that would require copying the local instrument and there may only be one local copy of the instrument (if you make a remote instrument, the server instance is the one local copy).


Parameter
~~~~~~~~~

.. Description

A Parameter represents a state variable of your system. Parameters may be settable, gettable, or both. Most of the time a Parameter is part of a single instrument_. These Parameters are created using ``instrument.add_parameter()`` and use the Instrument's low-level communication methods for execution. But a Parameter does not *need* to be part of an Instrument; Any object with ``get`` and/or ``set`` methods and a couple of descriptive attributes may be used as a Parameter.

A settable Parameter typically represents a configuration setting of an instrument, or some controlled characteristic of your system. If a settable Parameter is also gettable, getting it typically just reads back what was previously set (through QCoDeS or by some other means) but there can be differences due to rounding, clipping, feedback loops, etc. Note that a settable Parameter of a :ref:`metainstrument` may involve setting several lower-level Parameters of the underlying Instruments, or even getting the values of other Parameters to inform the value(s) to set.

A Parameter that is only gettable typically maps to a single measurement command or sequence. Often this returns a single value, like one voltage measurement, but it can also return multiple distinct values (for example magnitude and phase or the x/y/z components of a vector) or a sequence of values (for example an entire sampled waveform, or a power spectrum) or even multiple sequences (for example waveforms sampled on several channels).

Most Parameters that you would use as setpoints and measurements in a loop_ accept or return numbers, but configuration Parameters can use strings or any other data type (although it should generally be JSON-compatible for snapshots and logging).

When a RemoteInstrument is created, the Parameters contained in the Instrument are mirrored as RemoteParameters, which connect to the original Parameter via the associated InstrumentServer.

.. responsibilities
Parameters are responsible for:
  - (if part of an Instrument) generating the commands to pass to the Instrument and interpreting its response
  - (if not part of an Instrument) providing get and/or set methods
  - (if settable) testing whether an input is valid, usually using a :ref:`validator`.
  - providing context and meaning to the parameter through descriptive attributes:
    - name: its name as an attribute
    - label: short description as in an axis label
    - units: if the values are numbers, their units
    - and more if multi-valued or array-valued

.. state
Parameters hold onto their latest set or measured value, as well as when this happened, so that snapshots (and eventually monitor logs) need not always query the hardware for this information but can update it intelligently when it has gotten stale.

.. failures
A Parameter that is part of an Instrument, even though it can be used as an independent object without directly referencing the Instrument, is subject to the same local/remote limitations as the Instrument.

Loop
~~~~

.. Description

A Loop is the QCoDeS way to acquire one or more arrays of data. Every Loop that's executed consists of a settable Parameter to be varied, some collection of values to set it to, some actions to do at each setpoint, and some conditions by which to run the Loop. An action can be:
  - A gettable Parameter (something to measure). Each such Parameter will generate one (or more, if the Parameter itself creates multiple outputs).
  - A Task to do (for example you measure once, then have a Task to change a gate voltage, then you measure again, and finally a Task to put the gate voltage back where it was).
  - Wait, a specialized task that just delays execution (but may do other things like monitoring the system in that time)
  - BreakIf, a callable test for whether to quit (this level of) the Loop.
  - Another Loop nested inside the first, with all its own setpoints and actions. Measurements within a nested loop will produce a higher-dimension output array.

The key loop running conditions are:
  - background or not: A background Loop runs in its own separate process, so that you can be doing things in the main process like live plotting, analysis on the data as it arrives, preparing for the next measurement, or even unrelated tasks, while the Loop is running. The disadvantage is complexity, in that you can only use RemoteInstruments, and debugging gets much harder.
  - use threads: If true, we will group measurement actions and try to execute them concurrently across several threads. This can dramatically speed up slow measurements involving several instruments, but only if all instruments are in separate InstrumentServer processes, or all instruments are local.
  - data manager: If not False, we create another extra process whose job it is to offload data storage, and sync data back to the main process on demand, so that the Loop process can run with as little overhead as possible.
  - where and how to save the data to disk

.. responsibilities
The Loop is responsible for:
  - creating the dataset_ that will be needed to store its data
  - sequencing actions: the Loop should have the highest priority and the least overhead of extra responsibilities so that setpoints and actions occur with as fast and reliable timing as possible.

.. state
Before the Loop is run, it holds the setpoint and action definitions you are building up. You can actually keep a loop at any level of definition and reuse it later. Loop methods chain by creating entirely new objects, so that you can hold onto the Loop at any stage of definition and reuse just what has been defined up to that point.

After the Loop is run, it returns a dataset_ and the executed loop itself, along with the process it starts if it's a background Loop, only hold state (such as the current indices within the potentially nested Loops) while it is running.

.. failures
Loops can fail:
  - If you try to use a (parameter of a) local instrument in a background loop

DataSet
~~~~~~~

.. Description
.. responsibilities
.. state
.. failures
