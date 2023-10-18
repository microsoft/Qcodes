Benchmarking QCoDeS
===================

This directory contains benchmarks implemented for usage with airspeed
velocity ``asv`` benchmarking package. Refer to `asv documentation`_ for
information on how to use it.

.. _asv documentation: https://asv.readthedocs.io/en/stable/index.html

The ``asv.conf.json`` file in this directory configures ``asv`` to work
correctly with QCoDeS.

Usage
-----

Run benchmarking
````````````````

If you already have a python environment set up for QCoDeS, then run the
following command from this directory:

.. code:: bash

  asv run --python=same

If you do not have an environment set, then ``asv`` can set it up
automatically. The benchmarks are executed in the same way:

.. code:: bash

  asv run

Either of the commands above will execute benchmarking for the latest commit
of the main branch.

If you want to run benchmarking for a particular commit, use the same syntax
as there is used for ``git log`` (commit id with ``^!`` at the end; note that in
some terminals you will need to type ``^`` two times like this ``^^!``):

.. code:: bash

  asv run ed9b6fe8^!

Use the ``--bench`` option with a regular expression to tell ``asv`` which
benchmarks you would like to execute. For example, use the following syntax
to execute a benchmark called ``saving`` in ``data.py`` benchmark module:

.. code:: bash

  asv run --bench data.saving

Refer to `asv documentation`_ for more information on the various ways the
benchmarking can be executed (for example, how to run a particular
benchmark, how to compare results between commits, etc).

Display benchmarking results
````````````````````````````

In order to view the benchmarking results, execute the following command
to generate a convenient website

.. code:: bash

  asv publish

and the following command to start a simple server that could host the
website locally (the generated website is not static, hence the server is
needed)

.. code:: bash

  asv preview -b

The ``-b`` option opens the website automatically in your default browser
(the URL that it opens automatically is also printed to the terminal). In
order to stop the server, press ``Ctrl+C`` in the terminal where you've
started it.

Note that the benchmarking results are created locally on your machine, and
they get accumulated.

In order to compare benchmarking results of two commits, use the following
command (note that the benchmarking results for these two commits should
already exist):

.. code:: bash

  asv compare ed859c0a 8984aefb


Profile during benchmarking
```````````````````````````

If you would like to also profile while benchmarking in order to get more
insights on the performance of the code, use either

.. code:: bash

  asv run --profile

command or

.. code:: bash

  asv profile

command.

In case you would like to use a visualization tool for the profile results,
you can install one, for example, ``snakeviz``, and run benchmarking with
profiling as follows:

.. code:: bash

  asv profile --gui=snakeviz


ToDo for QCoDeS/core
--------------------

- host results and their html representation (GitHub pages?)
