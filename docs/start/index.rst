.. _gettingstarted:

Getting Started
===============

.. toctree::
   :maxdepth: 2


Requirements
------------

You need a working python 3.x installation to be able to use QCoDeS. We highly
recommend installing Anaconda, which takes care of installing Python and
managing packages. In the following it will be assumed that you use Anaconda.
Download and install it from `here <https://www.anaconda.com/download>`_. Make
sure to download the latest version with python 3.6.

Once you download, install Anaconda according to the instructions on screen,
choosing the single user installation option.

The next section will guide you through the installation of QCoDeS
on Windows, although most of the things also work for macOS and Linux.

Installation
------------
Before you install QCoDeS you have to decide whether you want to install the
latest stable release or if you want to get the developer version from GitHub.

Installing the latest QCoDeS release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
First download the QCoDeS environment.yml file by right
clicking on `this link <https://raw.githubusercontent.com/QCoDeS/Qcodes/master/environment.yml>`__
and select save link as and download the file to a location
that you can find again.
Next launch an Anaconda Prompt (start typing anaconda in the start menu and
click on *Anaconda Prompt*).

Here type in the prompt:

.. code:: bash

    conda env create -f environment.yml
    activate qcodes
    pip install qcodes

The first line creates a new anaconda environment that is called *qcodes*
and will install all the required dependendencies. The second line activates
this freshly created environment, so that the command in the third line will
install qcodes for this environment.

Installing QCoDeS from GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Clone the QCoDeS repository from GitHub from https://github.com/QCoDeS/Qcodes
Then create an environment, that contains all of the dependencies for QCoDeS,
from the *environment.yml* file in the root of the repository and activate it:

.. code:: bash

    conda env create -f <path-to-environment.yml>
    activate qcodes

Finally install QCoDeS add the repository via

.. code:: bash

    pip install -e <path-to-repository>

This will perform an `editable install` such that any changes of QCoDeS in your
local git clone are automatically available without having to reinstall QCoDeS.

Other dependencies
~~~~~~~~~~~~~~~~~~

You probably also wants to install National Instruments VISA from
`here <https://www.ni.com/visa/>`__. To download it
you will need to create an account on the National Instruments homepage but
the download is free of charge.

Updating QCoDeS
~~~~~~~~~~~~~~~

If you have installed with pip, run the following to update:

.. code:: bash

   pip install --upgrade qcodes

in principle, there should be a new release out roughly every month.

If you have installed with git, pull the QCoDeS repository using your
favourite method (git bash, git shell, github desktop, ...). There are
new commits to the repository daily.

Keeping your environment up to date
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Additional dependencies are periodically added to the QCoDeS environment and
new versions of packages that QCoDeS depends on are released.

To keep the QCoDeS environment up to date you can run

.. code:: bash

   conda update -n base conda -c defaults
   conda env update

from the root of your QCoDeS git repository.

Alternatively you can grab a new version of the ``environment.yml`` file as
explained above and run the same commands from a directory containing that file.

The first line ensures that the ``conda`` package manager it self is
up to date and the second line will ensure that the latest versions of the
packages used by QCoDeS are installed. See
`here <https://conda.io/docs/commands/env/conda-env-update.html>`__ for more
documentation on ``conda env update``.

If you using QCoDeS from an editable install you should also reinstall QCoDeS using

.. code:: bash

    pip install -e <path-to-repository>

After upgrading the environment to make sure that dependencies are tracked correctly.

Note that if you install packages yourself into the same
environment it is preferable to install them using ``conda``. There is a chance that
mixing packages from ``conda`` and ``pip`` will produce a broken environment.
Especially if the same package is installed using both ``pip`` and ``conda``.

Using QCoDes
------------
For using QCoDeS, as with any other python library, it is useful to use an
application that facilitates the editing and execution of python files. With
Anaconda come two preinstalled options:

 - **Jupyter**, a browser based notebook
 - **Spyder**, an integrated development environment

To start either of them you can use the shortcuts in the start menu under
*Anaconda3* with a trailing *(qcodes)*.

For other options you can launch a terminal either via the *Anaconda Navigator*
by selecting *qcodes* in the *Environments tab* and left-clicking on the *play*
button or by entering

.. code:: bash

    activate qcodes

in the *Anaconda prompt*

From the terminal you can then start any other application, such as *IPython* or
just plain old *Python*.


Getting started
---------------

Have a look at :ref:`userguide`, and or browse the examples at:

   https://github.com/QCoDeS/Qcodes/tree/master/docs/examples
