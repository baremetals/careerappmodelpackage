from career_app_model.predict import predict_suitability_score
from career_app_model.processing.prepare_data import json_to_dataframe


def test_make_prediction(test_sample_data):

    converted_data = json_to_dataframe(test_sample_data)

    # When
    result = predict_suitability_score(input_data=converted_data)

    # Then
    prediction = result.get("suitability_scores")
    assert isinstance(prediction, dict)
    assert result.get("errors") is None
    for key, value in result.items():
        assert isinstance(key, str)
        assert isinstance(key, float)
        assert 0 <= value <= 1
