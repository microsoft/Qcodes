# QCoDeS [![Build Status](https://travis-ci.com/qdev-dk/Qcodes.svg?token=H7MjHi74teZgv8JHTYhx&branch=master)](https://travis-ci.com/qdev-dk/Qcodes)

QCoDeS is a Python-based data acquisition framework developed by the Copenhagen / Delft / Sydney / Microsoft quantum computing consortium. While it has been developed to serve the needs of nanoelectronic device experiments, it is not inherently limited to such experiments, and can be used anywhere a system with many degrees of freedom is controllable by computer.

QCoDeS has taken inspiration from many similar frameworks that have come before it, including:
- [QTLab](https://github.com/heeres/qtlab)
- [Special Measure](https://github.com/yacobylab/special-measure)
- "Alex Igor Procedures" see [thesis](http://qdev.nbi.ku.dk/student_theses/pdf_files/A_Johnson_thesis.pdf) appendix D and [successors](http://www.igorexchange.com/project/Expt_Procedures)
- and countless smaller components created by students and postdocs throughout the quantum computing community

QCoDeS is compatible with Python 3.4+. It is primarily intended for use from Jupyter notebooks, but can be used from traditional terminal-based shells and in stand-alone scripts as well.
Although some feature at the moment are b0rken outside the notebook.

## Install

### PyPi
PyPi is the fastest way to install QCoDeS, will be avaiable once out of beta/private.

### Developer

We use virtualenv and pyenv to make sure all the system are the same, this rules out issues and the usual "it works on my machine".
Install virtual env (optionally virtualenvwrapper for convenience, if you are on linux) and pyenv according to your distribution.
Once all is installed, and working:

```bash
pyenv install 3.4.5
pyenv local 3.4.5
mkvirtualenv qcodes-dev -r develop_requirements.txt --python $HOME/.pyenv/versions/3.4.5/bin/python3.4
git clone https://github.com/qdev-dk/Qcodes.git $QCODES_INSTALL_DIR
cd $QCODES_INSTALL_DIR
pip install -r develop_requirements.txt
pip install -e .
python qcodes/test.py -f
```

If the tests pass you are ready to hack!
Note that sometimes the test suite because there is a bug somewhere in the mulitprocessing architecture.

This is the reference setup one needs to have to contribute, otherwise too many non-reproducible environments will show up.

### Anaconda

We recommend [Anaconda](https://www.continuum.io/downloads) as an easy way to get most of the dependencies out-of-the-box.

As the project is still private, install it directly from this repository:

- Install git: the [command-line toolset](https://git-scm.com/) is the most powerful but the [desktop GUI from github](https://desktop.github.com/) is also quite good

- Clone this repository somewhere on your hard drive. If you're using command line git, open a terminal window in the directory where you'd like to put QCoDeS and type:
```
git clone https://github.com/qdev-dk/Qcodes.git
```
#### the easy way
- Open the 'navigator' app that was installed with anaconda.
- On the left side click on "Environments".
- Then on the "import" icon, on the bottom.
- Pick a name, and click on the folder icon next to file to import from.
- Make sure you select "Pip requirement files" from the "Files of type" dialog then navigate to the qcodes folder and select `basic_requirements.txt`.
- Finally click import, and wait until done.
- The enviroment is now created, click on the green arrow to open a terminal inside it.
- Navigate again with the terminal (or drag and drop the the folder on OsX)
- Most likely you will want to plot stuff, so type:

  `conda install matplotlib`

  and after if you want qtplot

  `conda install pyqtgraph`

- Then type
  ` pip install -e . `


Finally bring Giulio  to Sabøtoren, Fensmarkgade 27, 2200 København N.


#### the not so easy way that often does not work

- Register qcoes  with Python, and install dependencies if any are missing: run this from the root directory of the repository you just cloned:
```
python setup.py develop
```

Now QCoDeS should be available to import into all Python sessions you run. To test, run `python` from some other directory (not where you just ran `setup.py`) and type `import qcodes`. If it works without an error you're ready to go.

### Plotting Requirements

Because these can sometimes be tricky to install (and not everyone will want all of them), the plotting packages are not set as required dependencies, so setup.py will not automatically install them. You can install them with `pip`:

- For `qcodes.MatPlot`: matplotlib version 1.5 or higher
- For `qcodes.QtPlot`: pyqtgraph version 0.9.10 or higher

### Updating QCoDeS

If you registered QCoDeS with Python via `setup.py develop`, all you need to do to get the latest code is open a terminal window pointing to anywhere inside the repository and run `git pull`

## Usage

Read the [docs](http://qdev-dk.github.io/Qcodes) and the notebooks in [docs/examples](docs/examples)


## Contributing

See [Contributing](CONTRIBUTING.rst) for information about bug/issue reports, contributing code, style, and testing
See the [Roadmap](http://qdev-dk.github.io/Qcodes/roadmap.html) an overview of where the project intends to go.


## Docs

We use sphinx for documentations, makefiles are provied boht for Windows, and *nix.

Go to the directory  `docs` and

```
make html
```

This generate a webpage, index.html,  in  `docs/_build/html` with the rendered html.
Documentation is updated  and deployed on every successful build.


## License

QCoDeS is currently a private development of Microsoft's Station Q collaboration, and IS NOT licensed for distribution outside the collaboration except by arrangement. We intend to release it as open source software once it is robust and reasonably stable, under the MIT license. See [License](LICENSE.md).
