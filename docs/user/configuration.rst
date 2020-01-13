Configuring QCoDeS
==================


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

.. jsonschema:: ../../qcodes/config/qcodesrc_schema.json


Using the config
----------------
QCoDeS exposes the default values in the module namespace as `defaults`

   >>> import qcodes
   >>> qcodes.defaults
   {'core': {'loglevel': 'DEBUG', 'legacy_mp': False}, 'user': {}, 'gui': {'notebook': True, 'plotlib': 'matplotlib'}}
   /home/unga/Hack/qdev/qcodes/config/config.json
   <qcodes.config.config.Config object at 0x7f920ec0eef0>

Values can be retrieved  by key :

>>> qcodes.defaults['user']
{}

Values can be retrieved by dotted keys:
>>> qcodes.defaults['gui.notebook']
True

Values can be described:

>>> qcodes.defaults.describe("gui")
"controls gui of qcodes.\n Current value: {'notebook': True, 'plotlib': 'matplotlib'}. Type: object. Default: Not defined."

qcodes.defaults looks like a dictionary (and supports all the dictionary operations you can think about), but it has some additional helpers.

Updating a value
~~~~~~~~~~~~~~~~
qcodes.defaults lets you insert a new value which gets stuffed in the "user" part of the config

>>> qcodes.defaults.add("foo", "bar")

>>> qcodes.defaults["user.foo"]
'bar'

And also pass a type validator  and a description:

>>> qcodes.defaults.add("foo", "bar", "string", "finds majorana")

>>> qcodes.defaults.describe("user.foo")
'finds majorana.
Current value: bar. Type: string. Default: Not defined.'

Saving
~~~~~~

If you made modifications you may want also to save them.
F.ex. this

>>> qcodes.defaults.save_to_home()

Will do what you think, i.e. saving to your home directory.
There is the possibility to save to env variable, and current working directory.

More
~~~~

For a overview of the API, read this: :ref:`config_api` .

.. note::  for now are supported all the JSON types MINUS enum
.. todo:: add GUI for creating config, explain saving (note on config loaded at module import so no effect if changed at runtime).

