from setuptools import setup, find_packages

import versioneer

def readme():
    with open('README.rst') as f:
        return f.read()


extras = {
    'MatPlot': {'matplotlib': '2.2.3'},
    'QtPlot': {'pyqtgraph': '0.11.0'},
    'coverage tests': {'coverage': '4.0'},
    'Slack': {'slacker': '0.9.42'},
    'ZurichInstruments': {'zhinst-qcodes': '0.1.1'},
    'test': {'pytest': '6.0.0',
             'PyVisa-sim': '0.4.0',
             'hypothesis': '5.0.0',
             'pytest-xdist': '2.0.0',
             'deepdiff': '5.0.2',
             'pytest-mock': "3.0.0",
             'pytest-rerunfailures': "5.0.0",
             'lxml': "4.3.0"
             }}

extras_require = {}
for extra_name, extra_packages in extras.items():
    extras_require[extra_name] = [f'{k}>={v}' for k, v in extra_packages.items()]


install_requires = [
    'numpy>=1.15',
    'pyvisa>=1.11.0, <1.12.0',
    'h5py>=2.8.0',
    'websockets>=7.0',
    'jsonschema>=3.0.0',
    'ruamel.yaml>=0.16.0,!=0.16.6',
    'wrapt>=1.10.4',
    'pandas>=0.24.0',
    'tabulate>=0.8.0',
    'tqdm>=4.20.0',
    'opencensus>=0.7.10, <0.8.0',
    'opencensus-ext-azure>=1.0.4, <2.0.0',
    'matplotlib>=2.2.3',
    "requirements-parser>=0.2.0",
    "importlib-metadata<4.0.0>1.0.0;python_version<'3.8'",
    "typing_extensions>=3.7.4 ",
    "packaging>=20.0",
    "ipywidgets>=7.5.0",
    "broadbean>=0.9.1",
]

setup(name='qcodes',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      use_2to3=False,

      maintainer='QCoDeS Core Developers',
      maintainer_email='qcodes-support@microsoft.com',
      description='Python-based data acquisition framework developed by the '
                  'Copenhagen / Delft / Sydney / Microsoft quantum computing '
                  'consortium',
      long_description=readme(),
      long_description_content_type='text/x-rst',
      url='https://github.com/QCoDeS/Qcodes',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3 :: Only',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering'
      ],
      license='MIT',
      # if we want to install without tests:
      # packages=find_packages(exclude=["*.tests", "tests"]),
      packages=find_packages(),
      package_data={'qcodes': ['monitor/dist/*', 'monitor/dist/js/*',
                               'monitor/dist/css/*', 'configuration/*.json',
                               'instrument/sims/*.yaml',
                               'tests/dataset/fixtures/2018-01-17/*/*',
                               'tests/drivers/auxiliary_files/*',
                               'py.typed', 'dist/schemas/*',
                               'dist/tests/station/*']},
      install_requires=install_requires,
      python_requires=">=3.7",
      test_suite='qcodes.tests',
      extras_require=extras_require,
      # zip_safe=False is required for mypy
      # https://mypy.readthedocs.io/en/latest/installed_packages.html#installed-packages
      zip_safe=False)
