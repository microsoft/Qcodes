.. _gettingstarted:

Getting Started
===============

.. toctree::
   :maxdepth: 2


Requirements
------------

For scientist we require to install anaconda, because it makes 
easy to get all dependencies. 
Download and install `here <https://www.continuum.io/downloads>`_.
Make sure to download the latest version with python 3.5.

Once you download, install anaconda according to the instructions on screen.

If you use *nix, you really should use the source.

The next section will guide you through the installation of qcodes
on windows, although most of the things also work for osx and Linux.
(that is, assuming you did not change your $PATH, and or have virtual envs,
but in that case you should once more just use the source)

Installation
------------
Once installed launch the anaconda navigator. 

-  On the left side click on "Environments".
-  Then on the "new" icon, on the bottom.
-  Select a name: for this example we use qcodes
-  Finally click "create", and wait until done.
-  The enviroment is now created, click on its name to activate it
-  Click the green arrow to open a terminal inside it.

Now you are ready to install QCoDeS, type in the terminal.

.. code:: bash

    conda install h5py
    conda install matplotlib
    pip install pyqtgraph
    pip install qcodes


It's nice to have both plotting libraries because pyqtgraph provides nice
interactive plots, and matplotlib publication ready figures (and also embedded
live updating plots in the notebook).


Enter QCoDes
------------
In  general follow this steps to get a terminal back:


- Open navigator
- On the left side click on "Environments".
- Click qcodes to activate it
- Click the green arrow to open a terminal inside it.

Now go to the directory of your experiment, and start a notebook.

.. code:: bash

	cd my_experiment
	jupyter notebook
	
Then:

  - build quantum computer
  - profit

Usage
-----

Read the :ref:`userguide`, and or browse the examples:

	https://github.com/QCoDeS/Qcodes/tree/master/docs/examples
