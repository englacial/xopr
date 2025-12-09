"""
Simple configuration management for STAC catalog generation using OmegaConf.

This module provides minimal configuration loading with OmegaConf,
letting the library handle all the complexity.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

from omegaconf import DictConfig, OmegaConf

# Register resolvers for common path expansions
OmegaConf.register_new_resolver("pwd", lambda: os.getcwd())
OmegaConf.register_new_resolver("home", lambda: Path.home())
OmegaConf.register_new_resolver("env", lambda x, default="": os.environ.get(x, default))


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[List[str]] = None,
    environment: Optional[str] = None
) -> DictConfig:
    """
    Load configuration from YAML file with optional overrides.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to YAML configuration file
    overrides : List[str], optional
        Command-line overrides in dot notation
        Example: ["data.primary_product=CSARP_qlook", "processing.n_workers=8"]
    environment : str, optional
        Environment name to apply (e.g., "production", "test", "development")

    Returns
    -------
    DictConfig
        Configuration object with dot-notation access

    Examples
    --------
    >>> conf = load_config("config/catalog.yaml")
    >>> print(conf.data.primary_product)
    'CSARP_standard'

    >>> conf = load_config(
    ...     "config/catalog.yaml",
    ...     overrides=["processing.n_workers=16"],
    ...     environment="production"
    ... )
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load user configuration directly
    conf = OmegaConf.load(config_path)

    # Apply environment-specific overrides if specified
    if environment and "environments" in conf:
        if environment in conf.environments:
            logging.info(f"Applying environment: {environment}")
            env_conf = conf.environments[environment]
            conf = OmegaConf.merge(conf, env_conf)
        else:
            logging.warning(f"Environment '{environment}' not found in config")

    # Apply command-line overrides
    if overrides:
        logging.debug(f"Applying overrides: {overrides}")
        override_conf = OmegaConf.from_dotlist(overrides)
        conf = OmegaConf.merge(conf, override_conf)

    # Resolve all interpolations (${...} references)
    OmegaConf.resolve(conf)

    # Remove environments section from runtime config (no longer needed)
    if "environments" in conf:
        del conf["environments"]

    return conf


def save_config(conf: DictConfig, output_path: Union[str, Path], add_metadata: bool = True):
    """
    Save configuration to file for reproducibility.

    Parameters
    ----------
    conf : DictConfig
        Configuration to save
    output_path : Union[str, Path]
        Where to save the configuration
    add_metadata : bool
        Whether to add generation metadata
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if add_metadata:
        # Add metadata about when/where this was generated
        from datetime import datetime
        save_conf = OmegaConf.create({
            "_metadata": {
                "generated_at": datetime.now().isoformat(),
                "working_directory": os.getcwd(),
            },
            **OmegaConf.to_container(conf)
        })
    else:
        save_conf = conf

    OmegaConf.save(save_conf, output_path)
    logging.info(f"Configuration saved to: {output_path}")


def validate_config(conf: DictConfig) -> bool:
    """
    Basic validation of required configuration fields.

    Parameters
    ----------
    conf : DictConfig
        Configuration to validate

    Returns
    -------
    bool
        True if valid, raises ValueError if not
    """
    required_fields = [
        "data.root",
        "data.primary_product",
        "output.path",
        "output.catalog_id",
        "output.catalog_description",
    ]

    for field in required_fields:
        if OmegaConf.select(conf, field) is None:
            raise ValueError(f"Required configuration field missing: {field}")

    # Validate n_workers is positive
    if conf.processing.n_workers <= 0:
        raise ValueError(f"Invalid n_workers: {conf.processing.n_workers}. Must be positive")

    # Validate paths exist
    data_root = Path(conf.data.root)
    if not data_root.exists():
        raise ValueError(f"Data root does not exist: {data_root}")

    return True


