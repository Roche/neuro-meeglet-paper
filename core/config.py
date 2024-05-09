"""
This module provides a Python object that contains the configurations from the config.yml file.
The config.yml file should be in the parent directory of this file.

Example usage:

from core.config import cfg
print(cfg['DATASETS']['TDBRAIN']['bids_root'])

"""

import yaml
from pathlib import Path

config_path = Path(__file__).parent / "../config.yml"

if not config_path.exists():
    raise FileNotFoundError(
        f"Configuration file does not exist. Please create the file by copying " +
        "the config.example.yml file and adjusting the configurations."
    )

with open(config_path, 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)