# utils/config_utils.py

import argparse
import yaml
from ast import literal_eval

def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--overrides', nargs='*', help='Override config entries. Format: key=value')
    args = parser.parse_args()

    override_dict = {}
    if args.overrides:
        for override in args.overrides:
            key, value = override.split('=', 1)
            try:
                # Attempt to evaluate the value as a Python literal (e.g., int, float, list, dict)
                _value = literal_eval(value)
            except:
                # If eval fails, keep it as a string
                _value = value
            override_dict[key] = _value

    return args.config, override_dict


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def merge_config(config: dict, overrides: dict) -> dict:
    for k, v in overrides.items():
        keys = k.split('.')
        d = config
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = v
    return config


def flatten_config(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
