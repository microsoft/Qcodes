QCoDeS is a python library that follows all the best practices of a python package.

All configuration for the package and for the tools we use lives in ``pyproject.toml``.

Documentation, in the ``docs/`` folder, is written using Restructured Text or Jupyter
Notebook formats, and sphinx is used to render the documentation into HTML pages to
be deployed at https://microsoft.github.io/Qcodes/.

The documentation includes Contributor's guide, so please follow that closely.

If the user manages the Python environment with ``uv``, always run commands that need that
environment (``pytest``, ``pyright``, ``python`` etc.) via ``uv run``. Both ``pytest`` and
``pyright`` need the test dependencies, so always run them with ``--extra test``, e.g.
``uv run --extra test pytest tests`` and ``uv run --extra test pyright``. Without it ``uv run``
only ensures that the core dependencies are installed, so the test dependencies may be missing.
On Windows, where ``pyright`` is typically installed as an npm shim, use
``uv run --extra test pyright.cmd``.
Do not activate the virtual environment, call executables inside it directly, or point tools at
an interpreter with options such as ``--pythonpath``; let ``uv run`` select the environment.

Before committing anything, run pre-commit hooks and make sure they pass, and fix anything that
is failing. If ``prek`` is available on the ``PATH``, use it via ``prek run --all-files``; otherwise
use ``pre-commit run --all``. The hooks will ensure correct formatting of the code and linting as
well. The hooks are automatically installed so there is no need to manually install them.

QCoDeS is a typed package, hence all new code should include clear and correct type annotations.

We use ``pyright`` to statically check the correctness of the code with type annotations.
Run it via ``pyright`` (or ``uv run --extra test pyright`` / ``uv run --extra test pyright.cmd``
as described above).
The code that should be typechecked is configured in ``pyproject.toml``.

For running tests, we use ``pytest``. They can be run with ``pytest tests``
(or ``uv run --extra test pytest tests`` as described above).
See pytest markers for additional options of running tests.

We use Dependabot from GitHub to keep our dependencies up to date, we use ``requirements.txt``
as our constraints or "lock" file.

Every Pull Request should have a newsfragment file briefly explaining the change. The text
should be using restructured text syntax. How to write a newsfragment and what types we
have is explained the Contributor's guide in the documentation in ``docs/`` folder.
