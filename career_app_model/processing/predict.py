import typing as t

import pandas as pd

from career_app_model import __version__ as _version
from career_app_model.config.core import config
from career_app_model.processing.data_manager import load_model
from career_app_model.processing.validations import validate_inputs

model_file_name = f"{config.app_config.model_save_file}{_version}.pkl"
_suitability_model = load_model(file_name=model_file_name)


def predict_suitability_score(*, input_data: t.Union[pd.DataFrame, dict], ) -> dict:
    """Predict the suitability score using a saved model."""

    user_data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=user_data)
    results = {"suitability_score": None, "version": _version, "errors": errors}

    if not errors:
        suitability = _suitability_model.predict(validated_data)
        results = {
            "suitability_score": suitability,
            "version": _version,
            "errors": errors
        }

    return results
