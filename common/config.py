#This module will read the CSV file and provide access to the configuration parameters.

import csv
import os

class ConfigError(Exception):
    pass

def load_config(file_path):
    config = {}
    if not os.path.exists(file_path):
        raise ConfigError(f"Configuration file {file_path} does not exist.")

    with open(file_path, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            if len(rows) != 2:
                raise ConfigError("Each row in the config file must have exactly two columns.")
            param, value = rows
            config[param.strip()] = value.strip()

    return config

def get_config_param(config, param_name, default_value=None, required=False):
    value = config.get(param_name)
    if required and not value:
        raise ConfigError(f"Required parameter '{param_name}' is missing in the config file.")
    return value if value else default_value
