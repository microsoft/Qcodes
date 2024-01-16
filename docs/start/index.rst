.. _gettingstarted:

Getting Started
===============

.. toctree::
   :maxdepth: 2


Requirements
------------

You need a working python 3.9 installation, as the minimum Python version, to be able to
use QCoDeS. We highly recommend installing Miniconda, which takes care of installing Python
and managing packages. In the following it will be assumed that you use Miniconda.
Download and install it from `here <https://docs.conda.io/en/latest/miniconda.html>`_. Make
sure to download the latest version with python 3.9 or newer.

Once you download, install Miniconda according to the instructions on screen,
choosing the single user installation option.

The next section will guide you through the installation of QCoDeS
on Windows, although most of the things also work for macOS and Linux.

Installation
------------
Before you install QCoDeS you have to decide whether you want to install the
latest stable release or if you want to get the developer version from GitHub.

Stable versions of QCoDeS are distributed via both PyPi and CondaForge to be installed
with pip and conda respectively. Below we will cover both installation types.
For new users we recommend installing QCoDeS from conda-forge.


Installing the latest QCoDeS release with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Next launch an Anaconda Prompt (start typing anaconda in the start menu and
click on *Anaconda Prompt*).

Here type in the prompt:

.. code:: bash

    conda create -n qcodes python=3.9
    conda activate qcodes
    pip install qcodes

The first line creates a new conda environment that is called *qcodes* with
python 3.9. The second line activates this freshly created environment, so that
the command in the third line will install qcodes for this environment.

Installing the latest QCoDeS release with conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install QCoDeS using Conda from conda-forge type the following into a prompt.


.. code:: bash

    conda create -n qcodes
    conda activate qcodes
    conda config --add channels conda-forge --env
    conda config --set channel_priority strict --env
    conda install qcodes

First we crate a new environment to install QCoDeS into and then we set that
environment up to install packages from conda-forge since QCoDeS is not available in the
default channel. As recommended by the conda-forge maintainers we set channel_priority to strict
to prefer all packages to be installed from the conda-forge channel.
Note that we use the `--env` flag to only set these settings for the QCoDeS env. We
do this to not change the settings for any other environment that you may have created.
Finally we install QCoDeS into this environment.



Installing QCoDeS from GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Clone the QCoDeS repository and submodules from GitHub from https://github.com/QCoDeS/Qcodes

.. code:: bash

    git clone --recurse-submodules https://github.com/QCoDeS/Qcodes <path-to-repository>

Finally install QCoDeS add the repository via

.. code:: bash

    pip install -e <path-to-repository>

This will perform an `editable install` such that any changes of QCoDeS in your
local git clone are automatically available without having to reinstall QCoDeS.

Note that if you wish to run the QCoDeS test you will also need to install the testing
dependencies. This can be done by installing QCoDeS using the `test` extra target.

.. code:: bash

    pip install -e <path-to-repository>[test]

Installing QCoDeS from a Forked GitHub Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you've forked the QCoDeS repository, make sure to also fetch the tags from the upstream repository for accurate versioning, especially if you wish to install with test dependencies. Not doing so may result in version conflicts.

.. code:: bash

    # Add the original QCoDeS repository as the 'upstream' remote
    git remote add upstream https://github.com/QCoDeS/Qcodes.git

    # Fetch all tags from the 'upstream' repository
    git fetch --tags upstream

After fetching the tags, proceed with the installation as usual:

.. code:: bash

    pip install -e <path-to-forked-repository>
    # Or with test dependencies
    pip install -e <path-to-forked-repository>[test]

Other dependencies
~~~~~~~~~~~~~~~~~~

To connect to many instruments (All instruments that are subclasses of
``VisaInstrument`` ) you need a working VISA implementation installed. There
are several of these available from instrument vendors and other sources.

We recommend you to install the Keysight IO Libraries Suite from `here
<https://www.keysight.com/find/iosuite>`__. To download it, you will need to
provide your e-mail id, name and location but the download is free of charge.

See the `PyVISA documentation <https://pyvisa.readthedocs
.io/en/latest/advanced/backends.html>`_ for more information.

Updating QCoDeS
~~~~~~~~~~~~~~~

If you have installed with pip, run the following to update:

.. code:: bash

   pip install --upgrade qcodes

If you have installed from conda-forge

.. code:: bash

   conda update qcodes

in principle, there should be a new release out roughly every month.

If you have installed with git, pull the QCoDeS repository using your
favourite method (git bash, git shell, github desktop, ...). There are
new commits to the repository daily.

Keeping your environment up to date
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Additional dependencies are periodically added to the QCoDeS and
new versions of packages that QCoDeS depends on are released.

Installing a new version of QCoDeS will automatically upgrade any dependency if required.
However you may wish to upgrade some of the dependency to a newer version than the minimum
version required by QCoDeS

If you have installed QCoDeS using pip you can upgrade each package individually.
E.g. to upgrade numpy you would do.

.. code:: bash

   pip install --upgrade numpy

You can list all outdated packages using

.. code:: bash

   pip list --outdated

If you have installed QCoDeS using conda upgrading all packages can be done by:

.. code:: bash

   conda update -n base conda -c defaults
   conda update --all

The first line ensures that the ``conda`` package manager it self is
up to date and the second line will ensure that the latest versions of the
packages used by QCoDeS are installed.

If you using QCoDeS from an editable install you should also reinstall QCoDeS using

.. code:: bash

    pip install -e <path-to-repository>

After upgrading the environment to make sure that dependencies are tracked correctly.

Note that if you install packages yourself into the same
environment it is preferable to install them using the same package manager that
you install QCoDeS with. There is a chance that
mixing packages from ``conda`` and ``pip`` will produce a broken environment.
Especially if the same package is installed using both ``pip`` and ``conda``.

Using QCoDes
------------
For using QCoDeS, as with any other python library, it is useful to use an
application that facilitates the editing and execution of python files. There
are two widely  used options:

 - **Jupyter**, a browser based notebook
 - **Spyder**, an integrated development environment

Both can be installed using ``conda`` or ``pip`` in the created ``conda`` environment
for QCoDeS. Then installation can be simply done by activating the QCoDeS environment
in the terminal and running the following:

.. code:: bash

    pip install spyder
    pip install jupyter

If you prefer *jupyterlab* over classic *jupyter notebook*, you should install it
separately:

.. code:: bash

    pip install jupyterlab

If you installed QCoDeS using conda-forge, you may want to install above using
``conda install`` rather than ``pip install``. After installation, they could be easily
called in your QCoDeS-activated ``conda`` terminal.

For running spyder:

.. code:: bash

    spyder

or for jupyter notebook:

.. code:: bash

    jupyter notebook

or jupyter lab:

.. code:: bash

    jupyter lab

For other options from the terminal you can activate the QCoDeS in that terminal
then start any other application, such as *IPython* or
just plain old *Python*.

It is also possible to install *Anaconda* for managing ``conda`` environments, however,
you should be mindful about using it as *Anaconda* installs a wide verity of packages in
the ``root`` environment upon installation. It comes with a *GUI* called *Anaconda Navigator*
to work with. If you rather to use this *GUI* over the ``conda`` terminal, make sure to select
the QCoDeS environment in the program, then you could be able to run the packages installed
in that environment. The reason is because by default, *Anaconda Navigator* starts with the
``base(root)`` environment.

Working example notebooks
-------------------------

For a more hands-on approach to learning about QCoDeS, have a look at `15 minutes to Qcodes <../examples/15_minutes_to_QCoDeS.ipynb>`__.

We also have a library of `examples notebooks <../examples/index.rst>`__ that detail more specific features of the software suite.
