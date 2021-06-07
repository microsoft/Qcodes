QCoDeS |PyPi| |DOCS| |PyPI python versions| |DOI|
=================================================
|Build Status Github| |Build Status Github Docs| |Codacy badge|

QCoDeS is a Python-based data acquisition framework developed by the
Copenhagen / Delft / Sydney / Microsoft quantum computing consortium.
While it has been developed to serve the needs of nanoelectronic device
experiments, it is not inherently limited to such experiments, and can
be used anywhere a system with many degrees of freedom is controllable
by computer.
To learn more about QCoDeS, browse our `homepage <http://qcodes.github.io/Qcodes>`_ .

To get a feeling of QCoDeS read
`15 minutes to QCoDeS <http://qcodes.github.io/Qcodes/examples/15_minutes_to_QCoDeS.html>`__,
and/or browse the Jupyter notebooks in `docs/examples
<https://github.com/QCoDeS/Qcodes/tree/master/docs/examples>`__ .

QCoDeS is compatible with Python 3.7+. It is primarily intended for use
from Jupyter notebooks, but can be used from traditional terminal-based
shells and in stand-alone scripts as well. The features in
`qcodes.utils.magic` are exclusively for Jupyter notebooks.


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

    make html

This generate a webpage, index.html, in ``docs/_build/html`` with the
rendered html.

Code of Conduct
===============

QCoDeS strictly adheres to the `Microsoft Open Source Code of Conduct <https://opensource.microsoft.com/codeofconduct/>`__


Contributing
============

The QCoDeS instrument drivers developed by the members of
the QCoDeS community but not supported by the QCoDeS developers are contained in

https://github.com/QCoDeS/Qcodes_contrib_drivers

See `Contributing <https://github.com/QCoDeS/Qcodes/tree/master/CONTRIBUTING.rst>`__ for general information about bug/issue
reports, contributing code, style, and testing.



License
=======

See `License <https://github.com/QCoDeS/Qcodes/tree/master/LICENSE.rst>`__.

.. |Build Status Github| image:: https://github.com/QCoDeS/Qcodes/workflows/Run%20mypy%20and%20pytest/badge.svg
    :target: https://github.com/QCoDeS/Qcodes/actions?query=workflow%3A%22Run+mypy+and+pytest%22
.. |Build Status Github Docs| image:: https://github.com/QCoDeS/Qcodes/workflows/build%20docs/badge.svg
    :target: https://github.com/QCoDeS/Qcodes/actions?query=workflow%3A%22build+docs%22
.. |Codacy badge| image:: https://api.codacy.com/project/badge/Grade/6c9e0e5712bf4c6285d6f717aa8e84fa
    :alt: Codacy Badge
    :target: https://app.codacy.com/manual/qcodes/Qcodes?utm_source=github.com&utm_medium=referral&utm_content=QCoDeS/Qcodes&utm_campaign=Badge_Grade_Settings
.. |PyPi| image:: https://badge.fury.io/py/qcodes.svg
    :target: https://badge.fury.io/py/qcodes
.. |PyPI python versions| image:: https://img.shields.io/pypi/pyversions/qcodes.svg
    :target: https://pypi.python.org/pypi/qcodes/
.. |DOCS| image:: https://img.shields.io/badge/read%20-thedocs-ff66b4.svg
   :target: http://qcodes.github.io/Qcodes
.. |DOI| image:: https://zenodo.org/badge/37137879.svg
   :target: https://zenodo.org/badge/latestdoi/37137879
