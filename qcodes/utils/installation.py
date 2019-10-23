"""This module contains helper scripts to make certain installation tasks
easier."""

import os
import json
from qcodes.station import SCHEMA_PATH, update_config_schema


def register_station_schema_with_vscode():
    """This function registeres the qcodes station schema with vscode.

    Run this function to add the user generated station schema to the list of
    associated schemas for the Red Hat YAML schema extension for vscode.
    (https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml)

    This function will effectively add an entry to `yaml.schemas` in the user
    config file of vscode, which is located under
    `%APPDATA/Code/User/settings.json`.

    You can alternatively access this
    setting via File->Preferences->Settings and search for `yaml.schemas`.

    To enable autocompletinon of QCoDeS instrument from additional packages run
    `qcodes.station.update_config_schema`.

    For more information consult `qcodes/docs/examples/Station.ipynb`.
    """
    if not os.path.exists(SCHEMA_PATH):
        update_config_schema()
    config_path = os.path.expandvars(
        os.path.join('%APPDATA%', 'Code', 'User', 'settings.json'))
    with open(config_path, 'r+') as f:
        data = json.load(f)
    data.setdefault(
        'yaml.schemas', {}
    )[r'file:\\' + os.path.splitdrive(SCHEMA_PATH)[1]] = '*.station.yaml'
    config_path_new = 'new_' + config_path
    with open(config_path_new, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(config_path, config_path_new)
