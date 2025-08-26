QCoDeS is a python library that follows all the best practices of a python package.

All configuration for the package and for the tools we use lives in ``pyproject.toml``.

Documentation, in the ``docs/`` folder, is written using Restructured Text or Jupyter
Notebook formats, and sphinx is used to render the documentation into HTML pages to
be deployed at https://microsoft.github.io/Qcodes/.

The documentation includes Contributor's guide, so please follow that closely.

Before committing anything, run pre-commit hooks with ```pre-commit run --all```
and make sure they pass, and fix anything that is failing. The pre-commit hooks will ensure
correct formatting of the code and linting as well. The pre-commit hooks are automatically installed
so there is no need to manually install them.

QCoDeS is a typed package, hence all new code should include clear and correct type annotations.

We use ``pyright`` to statically check the correctness of the code with type annotations.
Run it via ``pyright``. The code that should be typecheked is configured in ``pyproject.toml``.

For running tests, we use ``pytest``. They can be run with ``pytest tests``.
See pytest markers for additional options of running tests.

We use Dependabot from GitHub to keep our dependencies up to date, we use ``requirements.txt``
as our constraints or "lock" file.

Every Pull Request should have a newsfragment file briefly explaining the change. The text
should be using restructured text syntax. How to write a newsfragment and what types we
have is explained the Contributor's guide in the documentation in ``docs/`` folder.
