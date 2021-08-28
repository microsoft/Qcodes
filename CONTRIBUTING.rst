Contributing
============

Hi, thanks for your interest in the project! We welcome pull requests
from developers of all skill levels.

Jens H. Nielsen, William H.P Nielsen, Mikhail Astafev and Trevor Morgan
are the current maintainers of QCoDeS (aka core developers), along with a
group of talented and smart volunteers. Please don't hesitate to reach out at
qcodes-support@microsoft.com if you have any questions, or just need a
little help getting started.

Join us on Slack, where informal discussion is more than welcome.
(Please email us to be invited!)

.. contents::

QCoDeS Community Drivers
------------------------

The QCoDeS instrument drivers that are not supported by the QCoDeS developers
should be pushed to the dedicated repository in GitHub:

https://github.com/QCoDeS/Qcodes_contrib_drivers

These drivers are supported on a best effort basis by the developers of the individual drivers.

Note that, any pull request to the main QCoDeS repository concerning unsupported
drivers will not be reviewed and/or merged with the QCoDeS core.

Bugs reports and feature requests
---------------------------------

We use github's `issues <https://github.com/QCoDeS/Qcodes/issues>`__.
Search for existing and closed issues. If your problem or idea is not yet
addressed, `please open a new issue
<https://github.com/QCoDeS/Qcodes/issues/new>`__

The github GUI will show you a template both for bugs and features.
Delete the wrong part and try to follow the template. Writing a good
issue helps you in the first place. Bug reports must be accompanied by a
reproducible example.

Have an idea about future directions to go with Qcodes? Visions of
data-utopia that would take more than a few weeks to add or might change
some core ideas in the package? We can use issues for this too. We will pick the
``long-term`` or ``discussion`` labels.

 If somebody is assigned to an issue it means that somebody is working on it.

Clever usage
------------

Figured out a new way to use QCoDeS? Found a package that makes your
life better and easier? Got realtime analysis working after struggling
with it for days? Write it on Slack so we can keep github more
organized.

Development
-----------

Setup
~~~~~

-  Clone and register the package for development as described in the
   `README <README.md#installation>`__
-  Run tests
-  Ready to hack

.. _runnningtests:

Running Tests
~~~~~~~~~~~~~

We don't want to reinvent the wheel, and thus use py.test.
It's easy to install:

::

    pip install -r test_requirements.txt

Then to test and view the coverage:

::

    pytest --cov=qcodes --cov-report xml --cov-config=setup.cfg

To test and see the coverage (with missing lines) of a single module:

::

    pytest --cov=qcodes.module.submodule --cov-report=term-missing qcodes/tests/test_file.py

You can also run single tests with something like:

::

    pytest.exe .\qcodes\tests\test_config.py
    # or
    pytest.exe .\qcodes\tests\test_config.py::test_add_and_describe


If the tests pass, you should be ready to start developing!


New code and testing
~~~~~~~~~~~~~~~~~~~~
-  Fork the repo into your github account
-  Make a branch within this repo
-  It is worth considering a good branch name:

   -  for example selecting a prefix can be useful:

      -  feature/bar (if you add the feature bar)
      -  hotfix/bar (if you fix the bug bar)
      -  foo/bar (if you foo the bar)

   -  never use your username If you can't figure out a name for your
      branch, re-think about what you would be doing. It's always a good
      exercise to model the problem before you try to solve it. Also,
      ping on slack. We <3 you in the first place.


A note on committing and pushing (if you are not really familiar with git).
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A good commit is really important (for you writing it in the first
place). If you need a loving guide all the time you commit, see
`here <http://codeinthehole.com/writing/a-useful-template-for-commit-messages/>`__.
Do not push! Unless you are sure about your commits. If you have a typo
in your commit message, do not push. If you added more files/changes
that the commit says, do not push. In general everything is fixable if
you don't push. The reason is that on your local machine you can always
re-write history and make everything look nice, once pushed is just
harder to go back. If in doubt, ask and help will be given. Nobody was
born familiar with git, and everybody makes mistakes.

-  Write your new feature or fix. Be sure it doesn't break any existing
   tests, and please write tests that cover your feature as well, or if
   you are fixing a bug, write a test that would have failed before your
   fix. Our goal is 100% test coverage, and although we are not there,
   we should always strive to increase our coverage with each new
   feature. Please be aware also that 100% test coverage does NOT
   necessarily mean 100% logic coverage. If (as is often the case in
   Python) a single line of code can behave differently for different
   inputs, coverage in itself will not ensure that this is tested.

-  Write the docs, following the other documentation files (.rst) in the
   repo as an example. Or write the docs in the form of example IPython
   notebook (there are many of those in our docs as well).

-  We should have a *few* high-level "integration" tests, but simple
   unit tests (that just depend on code in one module) are more valuable
   for several reasons:
-  If complex tests fail it's more difficult to tell why
-  When features change it is likely that more tests will need to change
-  Unit tests can cover many scenarios much faster than integration
   tests.
-  If you're having difficulty making unit tests, first consider whether
   your code could be restructured to make it less dependent on other
   modules. Often, however, extra techniques are needed to break down a
   complex test into simpler ones. We are happy to help with this on Slack.
   Two ideas that are useful here:

   -  Patching, one of the most useful parts of the
      `unittest.mock <https://docs.python.org/3/library/unittest.mock.html>`__
      library. This lets you specify exactly how other functions/objects
      should behave when they're called by the code you are testing.
   -  Supporting files / data: Lets say you have a test of data acquisition
      and analysis. You can break that up into an acquisition test and an
      analysis by saving the intermediate state, namely the data file, in
      the test directory. Use it to compare to the output of the
      acquisition test, and as the input for the analysis test.

-  Refer to QCoDeS documentation on how to implement tests for the
   instrument drivers.

   -  We have not yet settled on a framework for testing real hardware.
      For some tests we use `pyvisa-sim <https://github.com/pyvisa/pyvisa-sim>`__
      but it's flexibility is limited. Another interesting candidate is
      `pyvisa-mock <https://github.com/microsoft/pyvisa-mock>`__.
      So, stay tuned, or post any ideas you have as GitHub issues!

Coding Style
~~~~~~~~~~~~

-  Try to make your code self-documenting. Python is generally quite
   amenable to that, but some things that can help are:

-  Use clearly-named variables
-  Only use "one-liners" like list comprehensions if they really fit on
   one line.
-  Comments should be for describing *why* you are doing something. If
   you feel you need a comment to explain *what* you are doing, the code
   could probably be rewritten more clearly.
-  If you *do* need a multiline statement, use implicit continuation
   (inside parentheses or brackets) and implicit string literal
   concatenation rather than backslash continuation
-  Format non-trivial comments using your GitHub nick and one of these
   prefixes:

   -  TODO( theBrain ): Take over the world!
   -  NOTE( pinky ): Well, that's a good idea.

-  Docstrings are required for modules, classes, attributes, methods, and
   functions (if public i.e no leading underscore). Because docstrings
   (and comments) *are not code*, pay special attention to them when
   modifying code: an incorrect comment or docstring is worse than none
   at all! Docstrings should utilize the `google
   style <http://google.github.io/styleguide/pyguide.html?showone=Comments#Comments>`__
   in order to make them read well, regardless of whether they are
   viewed through help() or on Read the Docs. See `the falcon
   framework <https://github.com/falconry/falcon>`__ for best practices
   examples.

-  Use `PEP8 <http://legacy.python.org/dev/peps/pep-0008/>`__ style. Not
   only is this style good for readability in an absolute sense, but
   consistent styling helps us all read each other's code.
-  There is a command-line tool (``pip install pycodestyle``) you can run after
   writing code to validate its style.
-  A lot of editors have plugins that will check this for you
   automatically as you type. Sublime Text for example has
   sublimelinter-pep8 and the even more powerful sublimelinter-flake8.
   For Emacs, the elpy package is strongly recommended (https://github.com/jorgenschaefer/elpy).
-  BUT: do not change someone else's code to make it pep8-compliant
   unless that code is fully tested.
-  BUT: remove all trailing spaces.
-  BUT: do not mix tabs and indentation for any reason.

-  JavaScript: The `Airbnb style
   guide <https://github.com/airbnb/javascript>`__ is quite good. If we
   start writing a lot more JavaScript we can go into more detail.

Pull requests
~~~~~~~~~~~~~

-  Push your branch back to github and make a pull request (PR). If you
   visit the repo `home page <ht://github.com/qcodes/Qcodes>`__ soon
   after pushing to a branch, github will automatically ask you if you
   want to make a PR and help you with it.

-  Naming matters; try to come up with a nice header:

   -  fix(dataformatter): Decouple foo from bar
   -  feature: Add logviewer

-  The template will help you write nice pull requests <3 !

-  Try to keep PRs small and focused on a single task. Frequent small
   PRs are much easier to review, and easier for others to work around,
   than large ones that touch the whole code base.

-  tag AT LEAST ONE person in the description of the PR (a tag is
   ``@username``) who you would like to have look at your work. Of
   course everyone is welcome and encouraged to chime in.

-  It's OK (in fact encouraged) to open a pull request when you still
   have some work to do. Just make a checklist
   (``- [ ] take over the world``) to let others know what more to
   expect in the near future.

-  Delete your branch once you have merged (using the helpful button
   provided by github after the merge) to keep the repository clean.
   Then on your own computer, after you merge and pull the merged master
   down, you can call ``git branch --merged`` to list branches that can
   be safely deleted, then ``git branch -d <branch-name>`` to delete it.

Automatic Testing (CI)
~~~~~~~~~~~~~~~~~~~~~~

Once your pull request is opened a number of automatic jobs are created. These
will run the tests and in other ways verify the correctness of the code.
In the following we will describe what we test and provide a few tips on how to
understand the results especially if something should fail.

Note that the some of the automatic jobs are labeled with Required. These
must pass before the pull request can be merged. The other jobs that do not
have a required label may be considered guidelines. Please attempt to make these
pass if possible but feel free to disregard them if the suggested changes do not make sense.
If in doubt feel free to ask questions.

Required checks
^^^^^^^^^^^^^^^

Our required checks consists of a number of jobs that performs the following actions using multiple python versions,
on Linux and on Windows.

- Run our test suite using pytest as described above.
- Perform type checking of the code in QCoDeS using MyPy. For many of the modules we enforce that the code must be
  type annotated. We encourage all contributors to type annotate any contribution to QCoDeS. If you need help with this
  please feel free to reach out.
- Build the documentation using Sphinx with Sphinx warnings as errors. This includes execution of all example notebooks
  that are not explicitly marked as not to be executed. Please see here_ for information on how to disable execution of a
  notebook.
- A number of smaller static checks implemented using `pre-commit <https://pre-commit.com/>`_ hooks. You may want to
  consider installing the pre-commit hooks in your local git config to have these checks performed automatically when
  you commit.

    - Check that YAML, JSON and Python files are syntactically valid.
    - Check that there are no trailing whitespace or blank lines at the end of python files.
    - Check that all files uses the correct line endings (``\n`` for all files except ``.bat``)
    - Run `pyupgrade  <https://github.com/asottile/pyupgrade>`_ on all python files.
    - Run `Darker <https://github.com/akaihola/darker/>`_. This will enforce `Black <https://github.com/psf/black>`_
      formatting and sorting of imports using `isort <https://pycqa.github.io/isort/>`_ on all new and changed code.
      We do not format the entire codebase to not lose change history.


Furthermore we also run our test suite with the minimum requirements stated to ensure that QCoDeS does work
correctly with these.

Optional checks
^^^^^^^^^^^^^^^

In addition to the required checks we perform two optional checks that can be regarded as guidelines rather than
requirements.

- We use Codacy to perform a number of style checks using `Pylint` and `Pydocstyle` among others. Please
  adapt your changes to these recommendations as you see fit. It is not a requirement that all Codacy warnings and
  errors are fixed. Do not insert comments to disable these warnings.
- We measure code coverage using `Codecov`. This measures if a line of code is executed as part of a test.
  As much as possible we would encourage you to add tests to cover all changes. However, this may not always be
  possible especially when writing instrument drivers.

Documenting QCoDeS
~~~~~~~~~~~~~~~~~~

All user facing modules should be included in the QCoDeS api documentation
on the QCoDeS homepage.

The documentation is generated by the ``.rst`` files in ``docs\api`` folder.
If you create a new user facing module you should take care to include it here.

For each folder of code there should be a matching folder in the ``docs\api``
folder containing an ``index.rst`` file and a ``X.rst`` file for each of
the ``X.py`` files that are to be documented. For instance assume that you want
to document ``qcodes.mymodule.a`` where ``mymodule`` is a folder containing an
``__init__.py`` and an ``a.py`` file. Then the ``mymodule`` folder within the ``api``
folder should contain an ``index.rst`` file and a ``a.rst`` file.

The ``index.rst`` file should then look like this::

    .. _mymodule :

    qcodes.mymodule
    ===============

    .. autosummary::

        qcodes.mymodule
        qcodes.mymodule.a


    .. automodule:: qcodes.mymodule


    .. toctree::
       :maxdepth: 4
       :hidden:

       a

This ``rst`` files will generate a page with the title ``qcodes.mymodule``.

The ``autosummary`` section generates a linked
table with the entries given. The ``automodule``
section generates the documentation for ``mymodule`` taken from the
``__init__.py`` file and the ``toctree`` section includes the doc pages of the
submodules that should be documented.

The submodule ``a.py`` is documented in its own file (``a.rst``) containing::

    qcodes.mymodule.a
    -----------------

    .. automodule:: qcodes.mymodule.a
       :members:

This automatically generates a page with the documentation of the module ``a.py``

Finally the ``index.rst`` file should be included in the toctree in ``docs/api/index.rst``

.. _here: ../examples/writing_drivers/Creating-Instrument-Drivers.ipynb
