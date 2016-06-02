QCodes FAQ
==========

This FAQ is intended for users for Qcodes. For development, see Github.

Installation
------------

How to install Qcodes?
~~~~~~~~~~~~~~~~~~~~~~

.. todo::  TBD

Usage
-----

Running measurement loops
~~~~~~~~~~~~~~~~~~~~~~~~~

``Loop.run()`` (or ``ActiveLoop.run()``) has a two arguments that
control where the loop runs and how the data is handled: 

``background``
    ``True`` (default) or ``False``

``data_manager``
    ``None`` (gets the default ``DataManager``) or ``False``


Loop modes of operations
~~~~~~~~~~~~~~~~~~~~~~~~

-  Default usage: ``Loop.run()``: Involves two extra processes, one
   ``<Measurement>`` process that sequences the loop actions and a
   ``<DataServer>`` process that the ``<Measurement>`` feeds data to,
   then stores it to disk and provides it to other processes that want
   it. This is the normal way to run loops, because it minimizes the
   work done in the measurement process, so it runs as fast as possible,
   and also keeps the main process free for other tasks like live
   plotting and analysis.

-  Foreground with a DataManager: ``Loop.run(background=False)``: The
   measurement loop runs in the process that started it, rather than
   making a new process, so the starting process blocks until the loop
   is finished. You might do this to make debugging easier.

-  Background with no DataManager: ``Loop.run(data_manager=False)``: The
   measurement loop runs in the background, but does not start (or
   connect to, if one is started already) a ``<DataServer>`` process;
   instead, it holds and stores the data itself. If the main process
   wants to sync this data during acquisition, it will need to read it
   from disk. Not sure why you would use this mode, but it's possible.

-  Foreground with no DataManager:
   ``Loop.run(background=False, data_manager=False)``: No extra
   processes are involved; the measurement loop runs in the process that
   started it, and holds and saves the data itself. If you want to start
   another measurement loop while one is already running (for example if
   you have a complex parameter that runs its own measurement loop to
   determine some derived value), you need to use this mode. That's
   because only one background measurement is allowed at a time, and
   only one ``DataSet`` may be on the ``<DataServer>`` at a time. But
   this mode will still save the ``DataSet`` it makes; in most such
   complex parameter cases you want an even more stripped-down loop:
   ``Loop.run_temp()`` which just calls:
   ``Loop.run(background=False, quiet=True, data_manager=False, location=False)``
   so it does not save anything, nor does it print the normal messages
   that ``run`` prints describing the ``DataSet`` it makes.

How to abort a running measurement ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``qc.halt_bg()``.

.. todo:: write what this does, which processes does it halts
          where data goes and so on.

.. todo::   does this apply for all the above modes?

List active loops
~~~~~~~~~~~~~~~~~
To list the active measurements use
``qc.active_children()``
