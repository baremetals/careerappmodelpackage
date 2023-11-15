import numpy as np
from career_app_model.predict import predict_suitability_scores


def test_make_prediction(test_sample_data):
    result = predict_suitability_scores(input_data=test_sample_data)
    prediction = result.get("suitability_scores")

    assert isinstance(prediction, np.ndarray), "prediction is not a NumPy array, it is None"
    assert result.get("errors") is None

    # Check each value in the prediction array
    for value in np.nditer(prediction[0]):
        value = float(value)
        assert isinstance(value, float), "Value in prediction is not a float"
        scaled_value = value / 10
        assert 0 <= scaled_value <= 2, "Value in prediction is not within the expected range when scaled"
