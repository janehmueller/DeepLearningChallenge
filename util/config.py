import json
from os import path

with open(path.join(path.dirname(__file__), '..', 'config.json')) as config_file:
    base_configuration = json.load(config_file)