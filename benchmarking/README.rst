Benchmarking QCoDeS
===================

This directory contains benchmarks implemented for usage with airspeed
velocity `asv` benchmarking package. Refer to `asv documentation`_ for
information on how to use it.

.. _asv documentation: https://asv.readthedocs.io/en/stable/index.html

The `asv.conf.json` file in this directory configures `asv` to work correctly
with QCoDeS.

Usage
-----

Run benchmarking
````````````````

If you already have a python environment set up for QCoDeS, then run the
following command from this directory:

.. code:: bash

  asv run python=same

If you do not have an environment set, then `asv` can set it up
automatically. The benchmarks are executed in the same way:

.. code:: bash

  asv run

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
website locally (the generated website is not static, hence the server)

  .. code:: bash

    asv preview

The `asv preview` command will print a URL that you can enter in your
browser to access the generated website.

Note that the benchmarking results are created locally on your machine, and
they get accumulated.

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
you can install one, for example, `snakeviz`, and run benchmarking with
profiling as follows:

.. code:: bash

  asv profile --gui=snakeviz


ToDo for QCoDeS/core
--------------------

- add benchmarking to CI
- host html results on GitHub pages


