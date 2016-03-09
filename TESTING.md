# Notes on test runners compatible with Qcodes

There is now a test script [test.py] in the root directory that uses the standard `unittest` machinery to run all the core tests (does not include instrument drivers). It has been tested on Mac (terminal), and Windows (cmd, git bash, and PowerShell). It includes coverage testing, but will only print a coverage report if tests pass.

The biggest difficulty with testing Qcodes is windows multiprocessing. The spawn method restricts execution in ways that are annoying for regular users (no class/function definitions in the notebook, no closures) but seem to be completely incompatible with some test runners (and/or coverage tracking)

I considered the following test runners:
- **nose**: works well, but it has a [note on its homepage](https://nose.readthedocs.org/en/latest/) that it is no longer being actively maintained (in favor of nose2 development), so we should not use it long-term.

- **unittest**: the standard, built-in python tester. The only thing we really need to add to this is coverage testing, so now the question is what's the easiest way to do this? On Windows just using unittest wrapped in coverage fails.

- **nose2**: has a broken coverage plugin - it reports all the unindented lines, ie everything
# that executes on import, as uncovered - but can be used by wrapping it inside coverage instead, just like unittest.

- **py.test**: seems to add lots of features, but it's not clear they are useful for us? Has a good coverage plugin but seems to require tons of command-line options. Requires both `pytest` and `pytest-cov` packages

on Mac terminal:
```
# the following work with coverage:
nosetests
python setup.py nosetests
py.test --cov-config .coveragerc --cov qcodes --cov-report term-missing
coverage run -m nose2 && coverage report -m
# both of these run unittest:
coverage run setup.py test && coverage report -m
coverage run -m unittest && coverage report -m

# nose2's coverage plugin is broken - it reports all the unindented lines (everything
# that executes on import) as uncovered
nose2 -C --coverage-report term-missing
```

Windows cmd shell and git bash behave identically, PowerShell has different chain syntax (commands with &&):
```
# the following work with coverage:
nosetests
py.test --cov-config .coveragerc --cov qcodes --cov-report term-missing
coverage run -m nose2 && coverage report -m  # cmd or bash
(coverage run -m nose2) -and (coverage report -m)  # PowerShell

# the following work without coverage:
python -m unittest discover
python -m unittest  # discover is unnecessary now, perhaps because I put test_suite in setup.py?

# the following do not work:

# fails on relative import in unittest inside separate process (why is it importing that anyway?)
coverage run -m unittest discover && coverage report -m  # cmd or bash
(coverage run -m unittest discover) -and (coverage report -m)  # PowerShell
# these fail asking for freeze_support() but nothing I do with that seems to help
python setup.py test
coverage run setup.py test && coverage report -m  # cmd or bash
(coverage run setup.py test) -and (coverage report -m)  # PowerShell
python setup.py nosetests
```
