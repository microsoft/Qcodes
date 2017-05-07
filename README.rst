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


QCoDeS is compatible with Python 3.5+. It is primarily intended for use
from Jupyter notebooks, but can be used from traditional terminal-based
shells and in stand-alone scripts as well. Although some feature at the
moment are b0rken outside the notebook.

Status
------
QCoDeS is still in development, more documentation and features will be coming!
The team behind this project just expanded.  There are still rough edges, and
gray areas but QCoDeS has been running without major issue in two long running
experiments.

The most important features in the roadmap are:

  - a robust architecture that uses the full potential of your harwdare
  - a more flexible and faster data storage solution

Install
=======

This is mostly for tech-y scientist, in general refer to `here <http://qcodes.github.io/Qcodes/start/index.html#installation>`__ 
for installation.

PyPi
----
.. code:: bash

    pip install qcodes

And see if you miss any dependencies.

Plotting Requirements
^^^^^^^^^^^^^^^^^^^^^^

Because these can sometimes be tricky to install (and not everyone will
want all of them), the plotting packages are not set as required
dependencies, so setup.py will not automatically install them. You can
install them with ``pip``:

-  For ``qcodes.MatPlot``: matplotlib version 1.5 or higher
-  For ``qcodes.QtPlot``: pyqtgraph version 0.9.10 or higher

Developer-pyenv
---------------

Core developers use virtualenv and pyenv to make sure all the system are the same,
this rules out issues and the usual "it works on my machine". Install pyenv
on your OS `see this <https://github.com/yyuu/pyenv>`__ .

$QCODES_INSTALL_DIR is the folder where you want to have the source code.

.. code:: bash

    git clone https://github.com/QCoDeS/Qcodes.git $QCODES_INSTALL_DIR
    cd $QCODES_INSTALL_DIR
    pyenv install 3.5.2
    pyenv virtualenv 3.5.2 qcodes-dev
    pyenv activate qcodes-dev
    pip install -r requirements.txt
    pip install coverage pytest-cov pytest --upgrade
    pip install -e .
    py.test --cov=qcodes --cov-config=.coveragerc

If the tests pass you are ready to hack!
This is the reference setup one needs to have to contribute, otherwise
too many non-reproducible environments will show up.

Updating QCoDeS
===============

from PyPi
---------

.. code:: bash

    pip install  --upgrade qcodes


Developer-pyenv/anaconda
------------------------

.. code:: bash

   cd $QCODES_INSTALL_DIR  && git pull


or if using GUIs, just pull the repo!


Docs
====

Read it `here <http://qcodes.github.io/Qcodes>`__ .
Documentation is updated and deployed on every successful build in master.


We use sphinx for documentations, makefiles are provided both for
Windows, and \*nix.

Go to the directory ``docs`` and

.. code:: bash

    make html

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
.. |DOI| image:: https://zenodo.org/badge/37137879.svg
   :target: https://zenodo.org/badge/latestdoi/37137879
