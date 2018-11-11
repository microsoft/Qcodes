# Legacy database file generation

## Why have old database files?

When the database schema (the database user version number) changes, the API in the core QCoDeS repository also changes to produce output matching the new version. That, in turn, makes the official API incapable of generating databases of older versions. To the end of testing the upgrade functions, we would like to be able to generate old version databases. The purpose of the `legacy_DB_generation` folder is contain scripts that generate old-version databases for the tests to consume.

When a new corner-case that we want to test for is discovered, we extend the scripts to produce a .db-file covering that case. When a new version upgrade is launched, we make a new script to generate .db-files for that version upgrade to test itself on.

## How to run the scripts?

The scripts use `gitpython` to roll QCoDeS back to a relevant old commit and create database files with the ancient source code.

The scripts should *not* be run as a part of the QCoDeS test suite, but prior to test execution in a different process. The scripts have some dependencies and will run in the normal QCoDeS environment **PROVIDED** that QCoDeS was installed with the editable flag (i.e. `pip install -e <path-to-qcodes>`).

## How do I write my own script?

First, please check if there is already a script producing .db-files of your desired version. If so, simply extend that script to produce a .db-file covering your particular test case. If not, follow this checklist:

 * Identify the database version of which you would like to create one or more .db-files. Let's call that "your version".
 * Check the variable `GIT_HASHES` in `utils.py` to see if "your version" already has a recorded commit hash.
   * If not, search through the `git log` of `master` to find the merge commit *just* before the merge commit that introduces the *next* version after "your version". Put that first commit into `GIT_HASHES` along with the version number of "your version".
 * Make a script called `generate_version_<your_version>.py`. Copy the general structure of `generate_version_0.py`. Make your generating functions take *ZERO* arguments and do all their imports inside their own scope.

## Anything else?

Remember to update the tests to use your newly generated fixtures. A test must **skip** (not fail) if the fixture is not present on disk. Also make sure that the CI runs your fixture-generating script.
