from setuptools import setup, find_packages
from distutils.version import StrictVersion
from importlib import import_module


def readme():
    with open('README.md') as f:
        return f.read()

extras = {
    'MatPlot': ('matplotlib', '1.5'),
    'QtPlot': ('pyqtgraph', '0.9.10'),
    'coverage tests': ('coverage', '4.0')
}
extras_require = {k: '>='.join(v) for k, v in extras.items()}

setup(name='qcodes',
      version='0.1.0',
      use_2to3=False,
      author='Alex Johnson',
      author_email='johnson.alex.c@gmail.com',
      maintainer='Alex Johnson',
      maintainer_email='johnson.alex.c@gmail.com',
      description='Python-based data acquisition framework developed by the '
                  'Copenhagen / Delft / Sydney / Microsoft quantum computing '
                  'consortium',
      long_description=readme(),
      url='https://github.com/qdev-dk/Qcodes',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3 :: Only',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Topic :: Scientific/Engineering'
      ],
      license='Private',
      # if we want to install without tests:
      # packages=find_packages(exclude=["*.tests", "tests"]),
      packages=find_packages(),
      package_data={'qcodes': ['widgets/*.js', 'widgets/*.css']},
      install_requires=[
          'numpy>=1.10',
          'pyvisa>=1.8',
          'IPython>=4.0',
          'ipywidgets>=4.1',
          # nose and coverage are only for tests, but we'd like to encourage
          # people to run tests!
          # coverage has a problem with setuptools on Windows, moved to extras
          'nose>=1.3'
      ],
      test_suite='qcodes.tests',
      extras_require=extras_require,
      # I think the only part of qcodes that would care about zip_safe
      # is utils.helpers.reload_code; users of a zip-installed package
      # shouldn't be needing to do this anyway, but we should test first.
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

# now test the versions of extras
for extra, (module_name, min_version) in extras.items():
    try:
        module = import_module(module_name)
        if StrictVersion(module.__version__) < StrictVersion(min_version):
            print(version_template.format(module_name, min_version, extra))
    except ImportError:
        print(missing_template.format(module_name, extra))
