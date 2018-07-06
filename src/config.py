import json
from os import path

with open(path.join(path.dirname(__file__), '..', 'config.json')) as config_file:
    base_configuration = json.load(config_file)

local_config_path = path.join(path.dirname(__file__), '..', 'config.local.json')
if path.exists(local_config_path):
    with open(local_config_path) as config_file:
        base_configuration = {**base_configuration, **json.load(config_file)}
