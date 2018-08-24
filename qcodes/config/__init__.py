import json

from .config import Config
from .config import logger
from .config import DotDict

cfg = Config()
loc = cfg['GUID_components']['location']
station = cfg['GUID_components']['work_station']

if station == 0 or loc == 0:
    print('Welcome to QCoDeS!\nTo use QCoDeS, please provide the system with '
          'enough information to generate unique dataset identifiers.')

if loc == 0:
    while loc == 0:
        loc = int(input('Please enter an integer as your location code: '))
    cfg['GUID_components']['location'] = loc

if station == 0:
    while station == 0:
        station = int(input('Please enter an integer as your work'
                            ' station code: '))
    cfg['GUID_components']['work_station'] = station

with open(cfg.default_file_name, "w") as fp:
            json.dump(cfg.current_config, fp, indent=4)
