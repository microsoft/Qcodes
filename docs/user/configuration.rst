Configuration
=============

QCoDeS uses a JSON based configuration system.
The configuration is modeled after the module structure plus some higher level switches.

Config files
------------
QCoDeS config files are read in the following order:

 - Package defaults
 - User's home directory
 - QCODES_CONFIG environment variable
 - Current working directory

This means that something specified in your current working directory will overwrite the package defaults.

Config files are backed by a json-schema, meaning that they will be validated before use, and if an error occurs the default settings are kept, and the user is informed.

.. note:: A change in the configuration requires reimporting the package, or restarting the notebook kernel.


Default config
--------------
The following  table explains the default configuration of qcodes.

.. jsonschema:: qcodes/config/schema.json

.. todo: add GUI for creating config, explain saving (note on config loaded at module import so no effect if changed at runtime).

