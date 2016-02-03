from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


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
          'ipywidgets>=4.1'
          # nose and coverage are only for tests, but we'd like to encourage
          # people to run tests!
          'nose>=1.3',
          'coverage>=4.0'
      ],
      extras_require={
          'MatPlot': ['matplotlib>=1.5'],
          'QtPlot': ['pyqtgraph>=0.9.10']
      },
      # I think the only part of qcodes that would care about zip_safe
      # is utils.helpers.reload_code; users of a zip-installed package
      # shouldn't be needing to do this anyway, but we should test first.
      zip_safe=False)
