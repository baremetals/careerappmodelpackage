import typing as t

from career_app_model import __version__ as _version
from career_app_model.config.core import config
from career_app_model.processing.data_manager import load_model, load_dataset
from career_app_model.processing.validations import validate_inputs
from career_app_model.processing.prepare_data import transform_input_for_prediction

model_file_name = f"{config.app_config.save_model_file}{_version}.pkl"
_suitability_model = load_model(file_name=model_file_name)
model_data = load_dataset(file_name=config.app_config.training_data_file)


def predict_suitability_scores(*, input_data: t.Dict) -> t.Dict:
    """Predict the suitability score using a saved model."""

    validated_data, errors = validate_inputs(input_data=input_data)
    results = {"suitability_scores": None, "version": _version, "errors": errors}

    if not errors:
        transformed_data = transform_input_for_prediction(input_data, model_data)

        suitability_scores = _suitability_model.predict(transformed_data)
        results = {
            "suitability_scores": suitability_scores,
            "version": _version,
            "errors": errors
        }
        print(results)

    return results
