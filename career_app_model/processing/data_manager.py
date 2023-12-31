import pickle
import typing as t
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from career_app_model import __version__ as _version
from career_app_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))

    return dataframe


def save_model(*, model_to_persist: RandomForestRegressor) -> None:
    """Persist the model.
        Saves the versioned model, and overwrites any previous
        saved models. This ensures that when the package is
        published, there is only one trained model that can be
        called, and we know exactly how it was built.
        """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.save_model_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_models(files_to_keep=[save_file_name])
    with open(save_path, 'wb') as file:
        pickle.dump(model_to_persist, file)


def load_model(*, file_name: str) -> RandomForestRegressor:
    """Load a persisted model."""

    file_path = TRAINED_MODEL_DIR / file_name
    with open(file_path, 'rb') as file:
        trained_model = pickle.load(file)
    return trained_model


def remove_old_models(*, files_to_keep: t.List[str]) -> None:
    """
        Remove old model pipelines.
        This is to ensure there is a simple one-to-one
        mapping between the package version and the model
        version to be imported and used by other applications.
        """
    do_not_delete = files_to_keep + ["__init__.py", "__pycache__"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            if model_file.is_dir():
                # Skip directories, or implement directory removal logic
                continue
            model_file.unlink()

