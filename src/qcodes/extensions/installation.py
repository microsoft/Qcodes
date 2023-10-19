"""This module contains helper scripts to make certain installation tasks
easier."""

import json
import os
import sys

from qcodes.station import SCHEMA_PATH, STATION_YAML_EXT, update_config_schema


def register_station_schema_with_vscode() -> None:
    """This function registeres the qcodes station schema with vscode.

    Run this function to add the user generated station schema to the list of
    associated schemas for the Red Hat YAML schema extension for vscode.
    (https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml)

    This function will effectively add an entry to `yaml.schemas` in the user
    config file of vscode, which is located under
    `%APPDATA/Code/User/settings.json`, or will be created there.

    You can alternatively access this
    setting via File->Preferences->Settings and search for `yaml.schemas`.

    To enable autocompletinon of QCoDeS instrument from additional packages run
    `qcodes.station.update_config_schema`.

    For more information consult `qcodes/docs/examples/Station.ipynb`.
    """
    if sys.platform != "win32":
        raise RuntimeError(
            "This script is only supported on Windows platforms.\n "
            "Please consult docstring for more information."
        )
    if not os.path.exists(SCHEMA_PATH):
        update_config_schema()

    config_path = os.path.expandvars(
        os.path.join("%APPDATA%", "Code", "User", "settings.json")
    )
    config_backup_path = config_path + "_backup"

    if not os.path.exists(config_path):
        raise RuntimeError(
            "Could not find the user settings file of vscode. \n"
            "Please refer to the station.ipynb notebook to learn how to "
            "set the settings manually."
        )
    with open(config_path, "r+") as f:
        data = json.load(f)
    data.setdefault("yaml.schemas", {})[
        r"file:\\" + os.path.splitdrive(SCHEMA_PATH)[1]
    ] = STATION_YAML_EXT

    os.replace(config_path, config_backup_path)

    with open(config_path, "w") as f:
        json.dump(data, f, indent=4)
