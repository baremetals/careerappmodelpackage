from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel
from strictyaml import YAML, load

import career_app_model

# Project Directories
PACKAGE_ROOT = Path(career_app_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    test_data_file: str
    save_model_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model training.
    """

    target: str
    features: List[str]
    test_size: float
    random_state: int
    n_estimators: int


class EmbeddingConfig(BaseModel):
    """
    All configuration relevant to creating embeddings
    locally.
    """
    embeddings_data_file: str
    embedding_dimension: int
    embedding_collection_name: str
    embedding_collection_description: str
    embedding_field_name: str
    embedding_role_id_name: str
    embedding_is_primary: bool
    embedding_max_length: int
    embedding_max_features: int
    embedding_group_by: str
    embedding_apply_to: str


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    app_model_config: ModelConfig
    embedding_config: EmbeddingConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        app_model_config=ModelConfig(**parsed_config.data),
        embedding_config=EmbeddingConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
