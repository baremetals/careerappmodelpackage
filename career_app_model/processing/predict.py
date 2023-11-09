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

    validated_data, errors = validate_inputs(input_data=input_data)
    results = {"suitability_scores": None, "version": _version, "errors": errors}

    if not errors:
        selected_industries = [
            input_data.get("selectedIndustry1"),
            input_data.get("selectedIndustry2"),
            input_data.get("selectedIndustry3"),
            input_data.get("selectedIndustry4"),
            input_data.get("selectedIndustry5")
        ]

        suitability_scores = {}
        for industry in selected_industries:
            # Using the actual input data for prediction
            predicted_score = _suitability_model.predict(input_data)[0]
            suitability_scores[industry] = predicted_score
        results = {
            "suitability_scores": suitability_scores,
            "version": _version,
            "errors": errors
        }

    return results
