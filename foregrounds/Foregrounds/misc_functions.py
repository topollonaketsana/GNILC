# Utility functions for configuration parsing
# Rewritten by ChatGPT (OpenAI) for Python 3 compatibility

import configparser

__all__ = ["ConfigSectionMap", "ConfigGetBoolean"]

def ConfigSectionMap(config, filename, section):
    """
    Parse a section of a .ini configuration file into a dictionary.

    Parameters:
    - config: a configparser.ConfigParser() instance
    - filename: path to the .ini file
    - section: section name to extract

    Returns:
    - dict of key-value pairs from the section
    """
    config.read(filename)
    if not config.has_section(section):
        raise ValueError(f"Section '{section}' not found in {filename}.")

    return {option: config.get(section, option) for option in config.options(section)}

def ConfigGetBoolean(config, filename, section, entry):
    """
    Extract a boolean entry from a .ini config section.

    Parameters:
    - config: a configparser.ConfigParser() instance
    - filename: path to the .ini file
    - section: section name
    - entry: specific boolean key

    Returns:
    - Boolean value of the key
    """
    config.read(filename)
    if not config.has_option(section, entry):
        raise ValueError(f"Option '{entry}' not found in section '{section}' of {filename}.")

    return config.getboolean(section, entry)

