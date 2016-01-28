# Qcodes

Qcodes is a Python-based data acquisition framework developed by the Copenhagen / Delft / Sydney / Microsoft quantum computing consortium. While it has been developed to serve the needs of nanoelectronic device experiments, it is not inherently limited to such experiments, and can be used anywhere a system with many degrees of freedom is controllable by computer.

Qcodes has taken inspiration from many similar frameworks that have come before it, including:
- [QTLab](https://github.com/heeres/qtlab)
- [Special Measure](https://github.com/yacobylab/special-measure)
- "Alex Igor Procedures" see [thesis](http://qdev.nbi.ku.dk/student_theses/pdf_files/A_Johnson_thesis.pdf) appendix D and [successors](http://www.igorexchange.com/project/Expt_Procedures)
- and countless smaller components created by students and postdocs throughout the quantum computing community

Qcodes is compatible with Python 3.3+. It is primarily intended for use from Jupyter notebooks, but can be used from traditional terminal-based shells and in stand-alone scripts as well.

## Installation

We recommend [Anaconda](https://www.continuum.io/downloads) as an easy way to get most of the dependencies out-of-the-box.

### Requirements

(Note: with the exception of Python itself, these are just the versions currently installed by the core developers. We haven't tested compatibility with lower versions of most packages, but if you know that an older version works fine, please let us know. If a lower OR higher version of any package breaks Qcodes please open an issue.)

- Python 3.3+
- numpy 1.10+
- pyvisa 1.8+
- IPython 4.0+
- ipywidgets 4.1+
- matplotlib 1.5+ (only for matplotlib plotting)
- pyqtgraph 0.9.10+ (only for pyqtgraph plotting)

for testing:

- nose 1.3+
- coverage 4.0+

## Usage

See the [docs](docs) directory, particularly the notebooks in [docs/examples](docs/examples)

Until we have this prepared as an installable package, you need to make sure Python can find qcodes by adding the repository root directory to `sys.path`:
```
import sys
qcpath = 'your/Qcodes/repository/path'
if qcpath not in sys.path:
    sys.path.append(qcpath)
```

## Contributing

See [Contributing](CONTRIBUTING.md) for information about bug/issue reports, contributing code, style, and testing

See the [Roadmap](ROADMAP.md) an overview of where the project intends to go.

## License

Qcodes is currently a private development of Microsoft's Station Q collaboration, and IS NOT licensed for distribution outside the collaboration. We intend to release it as open source software once it is robust and reasonably stable, under the MIT license or similar. See [License](LICENSE.txt)
