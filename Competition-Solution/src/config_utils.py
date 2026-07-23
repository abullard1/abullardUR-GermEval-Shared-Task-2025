# Competition-Solution/src/config_utils.py

"""
Configuration Management Utilities

This module provides utilities for loading and merging YAML configuration files
in the GermEval 2025 project. It drives the hierarchical config system where
experiment configs override subtask configs, which override base configs.

Author: Samuel Ruairí Bullard
Project: Evaluating Cost-Sensitive Loss Functions for Transformer-Based German Harmful Content Detection
Date: January 2026
"""

import yaml
import os
from pathlib import Path


# Recursively merges configuration dictionaries (override takes precedence)
def merge_configs(base, override):
    # Checks if the base is a dictionary
    if not isinstance(base, dict):
        return override
    
    # Checks if the override is a dictionary
    if not isinstance(override, dict):
        return override
    
    # Copies the base config
    merged = base.copy()
    
    # Merges the override config into the base config
    for key, value in override.items():
        # If the key is in the merged config and the value is a dictionary, merges the value
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        # If the key is not in the merged config or the value is not a dictionary, replaces the value
        else:
            merged[key] = value
    
    return merged


# Loads a YAML file and returns a dictionary.
def load_yaml(filepath):
    # Checks if the filepath is valid
    if not filepath or not os.path.exists(filepath):
        return {}
    
    # Tries to load the YAML file
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    # If the file is not a dictionary, returns an empty dictionary
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}


# Loads and merges the standard 3-level config hierarchy.
def load_config(base_config_path, subtask_config_path, experiment_config_path):
    # Loads all three config levels
    base = load_yaml(base_config_path)
    subtask = load_yaml(subtask_config_path) 
    experiment = load_yaml(experiment_config_path)
    
    # Merges in order (later configs override earlier ones)
    config = merge_configs(base, subtask)
    config = merge_configs(config, experiment)
    
    return config


# Helper to build standard config paths from the configs directory.
def get_config_paths(config_dir, subtask_name, experiment_name):
    # Gets the config paths
    config_dir = Path(config_dir)
    
    # Gets the base, subtask, and experiment paths
    base_path = config_dir / "base.yaml"
    subtask_path = config_dir / subtask_name / "base.yaml"
    experiment_path = config_dir / subtask_name / experiment_name
    
    # Returns the config paths
    return str(base_path), str(subtask_path), str(experiment_path)


# Validates that required keys exist in the config.
def validate_config(config, required_keys):
    # Initializes the missing keys list
    missing = []
    
    # Checks if the required keys exist in the config
    for key in required_keys:
        # Handles nested keys like "wandb.project"
        if "." in key:
            parts = key.split(".")
            current = config
            
            # Checks if the key is in the config
            for part in parts:
                # If the key is in the config, updates the current value
                if isinstance(current, dict) and part in current:
                    current = current[part]
                # If the key is not in the config, adds it to the missing keys list
                else:
                    missing.append(key)
                    break
        else:
            # If the key is not in the config, adds it to the missing keys list
            if key not in config:
                missing.append(key)
    
    # Returns if the config is valid and the missing keys
    return len(missing) == 0, missing


# Simplified all-in-one config loader for standard use cases.
def load_experiment_config(config_dir, subtask_name, experiment_name, required_keys=None):
    # Gets the standard config paths
    base_path, subtask_path, experiment_path = get_config_paths(
        config_dir, subtask_name, experiment_name
    )
    
    # Loads and merges the config
    config = load_config(base_path, subtask_path, experiment_path)
    
    # Validates the required keys
    if required_keys:
        is_valid, missing = validate_config(config, required_keys)
        # If the config is not valid, prints the missing keys
        if not is_valid:
            print(f"WARNING: Missing required config keys: {missing}")
    
    return config
