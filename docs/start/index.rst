.. _gettingstarted:

Getting Started
===============

.. toctree::
   :maxdepth: 2


Requirements
------------

For scientist we require to install Anaconda, because it makes
easy to get all dependencies. 
Download and install `here <https://www.anaconda.com/download>`_.
Make sure to download the latest version with python 3.6.

Once you download, install Anaconda according to the instructions on screen.

If you use \*nix, you really should use the source.

The next section will guide you through the installation of QCoDeS
on Windows, although most of the things also work for osx and Linux.
(that is, assuming you did not change your $PATH, and or have virtual envs,
but in that case you should once more just use the source)

Installation
------------

To setup an Anaconda environment for QCoDeS it's convenient to download
QCoDeS environment.yml file from
`here <https://github.com/QCoDeS/Qcodes/blob/master/environment.yml>`__.
Right click on raw and select save link as and download the file to a location
that you can find again.
Once Anaconda is installed and the environment.yml file downloaded
launch an Anaconda Prompt.

Now you are ready to install QCoDeS, type in the terminal.

.. code:: bash

    conda env create -f environment.yml
    pip install qcodes

Enter QCoDes
------------
In  general follow this steps to get a terminal back:


- Open navigator
- On the left side click on "Environments".
- Click qcodes to activate it
- Click the green arrow to open a terminal inside it.

Now go to the directory of your experiment, and start a notebook or spyder.

.. code:: bash

    cd my_experiment
    jupyter notebook

or

.. code:: bash

    cd my_experiment
    spyder

Then:

  - build quantum computer
  - profit

Usage
-----

Read the :ref:`userguide`, and or browse the examples:

   https://github.com/QCoDeS/Qcodes/tree/master/docs/examples
