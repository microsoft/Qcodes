QCoDeS |Build Status| |DOCS| |DOI|
===================================

QCoDeS is a Python-based data acquisition framework developed by the
Copenhagen / Delft / Sydney / Microsoft quantum computing consortium.
While it has been developed to serve the needs of nanoelectronic device
experiments, it is not inherently limited to such experiments, and can
be used anywhere a system with many degrees of freedom is controllable
by computer.
To learn more about QCoDeS, browse our `homepage <http://qcodes.github.io/Qcodes>`_ .

To get  a feeling of qcodes browse the Jupyter notebooks in `docs/examples
<https://github.com/QCoDeS/Qcodes/tree/master/docs/examples>`__ .


QCoDeS is compatible with Python 3.6+. It is primarily intended for use
from Jupyter notebooks, but can be used from traditional terminal-based
shells and in stand-alone scripts as well. The features in
`qcodes.utils.magic` are exclusively for Jupyter notebooks.

Status
------
QCoDeS is still in development, more documentation and features will be coming!
The team behind this project just expanded.  There are still rough edges, and
gray areas but QCoDeS has been running without major issue in two long running
experiments.

The most important features in the roadmap are:

- a more flexible and faster data storage solution
- a robust architecture that uses the full potential of your hardware


Install
=======

In general, refer to `here <http://qcodes.github.io/Qcodes/start/index.html#installation>`__
for installation.


Docs
====

Read it `here <http://qcodes.github.io/Qcodes>`__ .
Documentation is updated and deployed on every successful build in master.

We use sphinx for documentations, makefiles are provided both for
Windows, and \*nix, so that you can build the documentation locally.

Make sure that you have the extra dependencies required to install the docs

.. code:: bash

    pip install -r docs_requirements.txt

Go to the directory ``docs`` and

.. code:: bash

    make html-api

This generate a webpage, index.html, in ``docs/_build/html`` with the
rendered html.

Contributing
============

See `Contributing <https://github.com/QCoDeS/Qcodes/tree/master/CONTRIBUTING.rst>`__ for information about bug/issue
reports, contributing code, style, and testing



License
=======

See `License <https://github.com/QCoDeS/Qcodes/tree/master/LICENSE.rst>`__.

.. |Build Status| image:: https://travis-ci.org/QCoDeS/Qcodes.svg?branch=master
    :target: https://travis-ci.org/QCoDeS/Qcodes
.. |DOCS| image:: https://img.shields.io/badge/read%20-thedocs-ff66b4.svg
   :target: http://qcodes.github.io/Qcodes
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3233612.svg
   :target: https://doi.org/10.5281/zenodo.3233612
