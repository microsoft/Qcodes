---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# QCoDeS config

The QCoDeS config module uses JSON files to store QCoDeS configuration modeled after the module structure with some high-level switches. 

The config file controls various options to QCoDeS such as the default path and name of the database in which your data is stored and logging level of the debug output. QCoDeS is shipped with a default configuration. As we describe below, you may overwrite these default values to customize QCoDeS for your nanoelectronics research. 

You may want to do something as simple as changing the default path of your database to something complex as including your own configuration variables. In this example notebook, we will explore some of the configuration variables, demonstrate how configuration values may be changed at runtime and saved to files, and explore using custom configuration values to suit your own needs.   

+++

QCoDeS loads both the defaults and the active configuration at the module import so that you can directly inspect them

```{code-cell} ipython3
import qcodes as qc
```

```{code-cell} ipython3
qc.config.current_config
```

The current config is build from multiple config files.

The config files are read in the following order:

* Package defaults
* User's home directory
* QCODES_CONFIG environment variable
* Current working directory

Meaning that any value set in a config file in the package default is overwritten by the same value in the users home dir and so on. 

Note: A change in the configuration requires reimporting the package, or restarting the notebook kernel.

```{code-cell} ipython3
qc.config.defaults
```

One can inspect what the configuration options mean at runtime

```{code-cell} ipython3
print(qc.config.describe('core'))
```

The default configuration is explained in more detail in the [api docs](../api/configuration/index.rst#qcodes-default-configuration)

+++

## Configuring QCoDeS

+++

Defaults are the settings that are shipped with the package, which you can overwrite programmatically.

+++

A way to customize QCoDeS is to write your own JSON files, they are expected to be in the directories printed below and documented above.
One will be empty because one needs to define first the environment variable in the OS. 

They are ordered by "weight", meaning that the last file always wins if it's overwriting any preconfigured defaults or values in the other files.

Simply copy the file to the directories and you are good to go.

```{code-cell} ipython3
print("\n".join([qc.config.home_file_name, qc.config.env_file_name, qc.config.cwd_file_name]))
```

The easiest way to add something to the configuration is to use the provided helper:

```{code-cell} ipython3
qc.config.add("base_location", "/dev/random", value_type="string", description="Location of data", default="/dev/random")
```

This will add a `base_location` with value `/dev/random` to the current configuration, and validate it's value to be of type string, will also set the description and what one would want to have as default.
The new entry is saved in the 'user' part of the configuration.

```{code-cell} ipython3
print(qc.config.describe('user.base_location'))
```

You can also manually update the configuration from a specific file by supplying the path of the directory as the argument of `qc.config.update_config` method as follows: 

```{code-cell} ipython3
qc.config.update_config(path="C:\\Users\\jenielse\\")
```

## Saving changes

+++

All the changes made to the defaults are stored, and one can then decide to save them to the desired location.

```{code-cell} ipython3
help(qc.config.save_to_cwd)
```

```{code-cell} ipython3
help(qc.config.save_to_env)
```

```{code-cell} ipython3
help(qc.config.save_to_home)
```

### Using custom variables in your experiment:

Simply get the value you have set before with dot notation.
For example:

```{code-cell} ipython3
qc.config.add("base_location", "/dev/random", value_type="string", description="Location of data", default="/dev/random")

qc.config.current_config
```

## Changing core values

+++

One can change the core values at runtime, but there is no guarantee that they are going to be valid.
Since user configuration shadows the default one that comes with QCoDeS, apply care when changing the values under `core` section. This section is, primarily, meant for the settings that are determined by QCoDeS core developers. 

```{code-cell} ipython3
qc.config.current_config.core.loglevel = 'INFO'
```

But one can manually validate via 

```{code-cell} ipython3
qc.config.validate()
```

Which will raise an exception in case of bad inputs

```{code-cell} ipython3
qc.config.current_config.core.loglevel = 'YOLO'
qc.config.validate()
# NOTE that you how have a broken config! 
```
