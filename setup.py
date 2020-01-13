from setuptools import setup, find_packages
from distutils.version import StrictVersion
from importlib import import_module
import sys

import versioneer

def readme():
    with open('README.rst') as f:
        return f.read()


extras = {
    'MatPlot': ('matplotlib', '2.2.3'),
    'QtPlot': ('pyqtgraph', '0.10.0'),
    'coverage tests': ('coverage', '4.0'),
    'Slack': ('slacker', '0.9.42')
}
extras_require = {k: '>='.join(v) for k, v in extras.items()}

install_requires = [
    'numpy>=1.10',
    'pyvisa>=1.9.1',
    'h5py>=2.6',
    'websockets>=7.0',
    'jsonschema',
    'ruamel.yaml',
    'pyzmq',
    'wrapt',
    'pandas',
    'tabulate',
    'tqdm',
    'applicationinsights',
    'matplotlib>=2.2.3',
    "dataclasses;python_version<'3.7'",  # can be removed once we drop support for python 3.6
    "requirements-parser",
    "importlib-metadata;python_version<'3.8'"
]

setup(name='qcodes',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      use_2to3=False,

      maintainer='Jens H Nielsen',
      maintainer_email='Jens.Nielsen@microsoft.com',
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
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering'
      ],
      license='MIT',
      # if we want to install without tests:
      # packages=find_packages(exclude=["*.tests", "tests"]),
      packages=find_packages(),
      package_data={'qcodes': ['monitor/dist/*', 'monitor/dist/js/*',
                               'monitor/dist/css/*', 'config/*.json',
                               'instrument/sims/*.yaml',
                               'tests/dataset/fixtures/2018-01-17/*/*',
                               'tests/drivers/auxiliary_files/*',
                               'py.typed', 'dist/schemas/*',
                               'dist/tests/station/*']},
      install_requires=install_requires,

      test_suite='qcodes.tests',
      extras_require=extras_require,
      # zip_safe=False is required for mypy
      # https://mypy.readthedocs.io/en/latest/installed_packages.html#installed-packages
      zip_safe=False)

version_template = '''
*****
***** package {0} must be at least version {1}.
***** Please upgrade it (pip install -U {0} or conda install {0})
***** in order to use {2}
*****
'''

missing_template = '''
*****
***** package {0} not found
***** Please install it (pip install {0} or conda install {0})
***** in order to use {1}
*****
'''

valueerror_template = '''
*****
***** package {0} version not understood
***** Please make sure the installed version ({1})
***** is compatible with the minimum required version ({2})
***** in order to use {3}
*****
'''

othererror_template = '''
*****
***** could not import package {0}. Please try importing it from
***** the commandline to diagnose the issue.
*****
'''

# now test the versions of extras
for extra, (module_name, min_version) in extras.items():
    try:
        module = import_module(module_name)
        if StrictVersion(module.__version__) < StrictVersion(min_version):
            print(version_template.format(module_name, min_version, extra))
    except ImportError:
        print(missing_template.format(module_name, extra))
    except ValueError:
        print(valueerror_template.format(
            module_name, module.__version__, min_version, extra))
    except:
        print(othererror_template.format(module_name))
