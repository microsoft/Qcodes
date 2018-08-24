# This script is to be run by CI to overwrite the user-specific settings in
# the config file

import os
import json
import logging

log = logging.getLogger(__name__)

path_to_script = str(os.path.realpath(__file__))
path_to_cfg = os.sep.join(path_to_script.split(os.sep)[:-2] +
                          ['config', 'qcodesrc.json'])
with open(path_to_cfg, 'r') as f:
    cfg = json.loads(f.read())

# for safety, only modify the file if we have to
if cfg['GUID_components']['location'] == 0:
    mock_location = 1
    cfg['GUID_components']['location'] = mock_location
    log.info(f'Found location to be 0. Changing it to {mock_location}.')

if cfg['GUID_components']['work_station'] == 0:
    mock_station = 1
    cfg['GUID_components']['work_station'] = mock_station
    log.info(f'Found work station to be 0. Changing it to {mock_station}')

with open(path_to_cfg, 'w') as f:
    json.dump(cfg, f, indent=4)
