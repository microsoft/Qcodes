QCoDeS |PyPi| |DOCS| |PyPI python versions| |DOI|
=================================================
|Build Status Github| |Build Status Github Docs| |Ruff| |OpenSSF|

QCoDeS is a Python-based data acquisition framework developed by the
Copenhagen / Delft / Sydney / Microsoft quantum computing consortium.
While it has been developed to serve the needs of nanoelectronic device
experiments, it is not inherently limited to such experiments, and can
be used anywhere a system with many degrees of freedom is controllable
by computer.
To learn more about QCoDeS, browse our `homepage <http://microsoft.github.io/Qcodes>`_ .

To get a feeling of QCoDeS read
`15 minutes to QCoDeS <http://microsoft.github.io/Qcodes/examples/15_minutes_to_QCoDeS.html>`__,
and/or browse the Jupyter notebooks in `docs/examples
<https://github.com/QCoDeS/Qcodes/tree/main/docs/examples>`__ .

QCoDeS is compatible with Python 3.10+. It is
primarily intended for use from Jupyter notebooks, but can be used from
traditional terminal-based shells and in stand-alone scripts as well. The
features in `qcodes.utils.magic` are exclusively for Jupyter notebooks.


Default branch is now main
==========================

The default branch in QCoDeS has been renamed to main.
If you are working with a local clone of QCoDeS you should update it as follows:

* Run `git fetch origin` and `git checkout main`
* Run `git symbolic-ref refs/remotes/origin/HEAD refs/remotes/origin/main` to update your HEAD reference.

Install
=======

In general, refer to `here <http://microsoft.github.io/Qcodes/start/index.html#installation>`__
for installation.


Docs
====

Read it `here <http://microsoft.github.io/Qcodes>`__ .
Documentation is updated and deployed on every successful build in main.

We use sphinx for documentations, makefiles are provided both for
Windows, and \*nix, so that you can build the documentation locally.

Make sure that you have the extra dependencies required to install the docs

.. code:: bash

    pip install -r docs_requirements.txt

Go to the directory ``docs`` and

.. code:: bash

    make html

This generate a webpage, index.html, in ``docs/_build/html`` with the
rendered html.

QCoDeS Loop
===========

The modules ``qcodes.data``, ``qcodes.plots``, ``qcodes.actions``,
``qcodes.loops``, ``qcodes.measure``, ``qcodes.extensions.slack``
and ``qcodes.utils.magic`` that were part of QCoDeS until version 0.37.0.
have been moved into an independent package called qcodes_loop.
Please see it's `repository <https://github.com/QCoDeS/Qcodes_loop/>`_ and
`documentation <https://microsoft.github.io/Qcodes_loop/>`_ for more information.

For the time being it is possible to automatically install the qcodes_loop
package when installing qcodes by executing ``pip install qcodes[loop]``.

Code of Conduct
===============

QCoDeS strictly adheres to the `Microsoft Open Source Code of Conduct <https://opensource.microsoft.com/codeofconduct/>`__


Contributing
============

The QCoDeS instrument drivers developed by the members of
the QCoDeS community but not supported by the QCoDeS developers are contained in

https://github.com/QCoDeS/Qcodes_contrib_drivers

See `Contributing <https://github.com/QCoDeS/Qcodes/tree/main/CONTRIBUTING.rst>`__ for general information about bug/issue
reports, contributing code, style, and testing.

Trademarks
==========

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or
logos is subject to and must follow Microsoft’s Trademark & Brand Guidelines. Use of Microsoft trademarks
or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party’s policies.

License
=======

See `License <https://github.com/QCoDeS/Qcodes/tree/main/LICENSE>`__.

.. |Build Status Github| image:: https://github.com/QCoDeS/Qcodes/workflows/Run%20mypy%20and%20pytest/badge.svg
    :target: https://github.com/QCoDeS/Qcodes/actions?query=workflow%3A%22Run+mypy+and+pytest%22+branch%3Amain
.. |Build Status Github Docs| image:: https://github.com/QCoDeS/Qcodes/workflows/build%20docs/badge.svg
    :target: https://github.com/QCoDeS/Qcodes/actions?query=workflow%3A%22build+docs%22+branch%3Amain
.. |Ruff|  image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff
.. |PyPi| image:: https://badge.fury.io/py/qcodes.svg
    :target: https://badge.fury.io/py/qcodes
.. |PyPI python versions| image:: https://img.shields.io/pypi/pyversions/qcodes.svg
    :target: https://pypi.python.org/pypi/qcodes/
.. |DOCS| image:: https://img.shields.io/badge/read%20-thedocs-ff66b4.svg
   :target: http://microsoft.github.io/Qcodes
.. |DOI| image:: https://zenodo.org/badge/37137879.svg
   :target: https://zenodo.org/badge/latestdoi/37137879
.. |OpenSSF| image:: https://api.securityscorecards.dev/projects/github.com/microsoft/Qcodes/badge
   :target: https://securityscorecards.dev/viewer/?uri=github.com/microsoft/Qcodes
