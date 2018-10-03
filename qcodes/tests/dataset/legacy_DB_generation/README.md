# Why a legacy folder?

When the database schema (the database user version number) changes, the API in the core QCoDeS repository also changes to produce output matching the new version. That, in turn, makes the official API incapable of generating databases of older versions. To the end of testing the upgrade functions, we would like to be able to generate old version databases. The purpose of the `legacy_DB_generation` folder is contain scripts that generate old-version databases for the tests to consume.

The scripts use `gitpython` to roll QCoDeS back to a relevant old commit and create database files with the ancient source code.

The scripts should *not* be run as a part of the QCoDeS test suite, but prior to test execution in a different process. The scripts have some dependencies and will run in the normal QCoDeS environment **PROVIDED** that QCoDeS was installed with the editable flag (i.e. `pip install -e <path-to-qcodes>`).
