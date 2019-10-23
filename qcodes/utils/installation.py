import os
import json
from qcodes.station import SCHEMA_PATH


def register_station_schema_with_vscode():
    config_path = os.path.expandvars(os.path.join('%APPDATA%', 'Code', 'User', 'settings.json'))
    with open(config_path, 'r+') as f:
        data = json.load(f)
        # "file:///Users/a-dovoge/Qcodes/qcodes/dist/schemas/station-template.schema.json": "*.station.yaml"
    _, schema_path = os.path.splitdrive(SCHEMA_PATH)
    data.setdefault('yaml.schemas', {})[r'file:\\' + schema_path] = '*.station.yaml'
    os.remove(config_path)
    with open(config_path, 'w') as f:
        json.dump(data, f, indent=4)
