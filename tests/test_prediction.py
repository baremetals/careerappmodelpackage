from career_app_model.processing.predict import predict_suitability_score
from career_app_model.processing.prepare_data import json_to_dataframe


def test_make_prediction(user_sample_input_data):
    # Given Sample valid input data from the Node.js server

    input_data = {
        "userId": "64b16ff9746c9b729c5",
        "selectedIndustries": ["Banking and Finance", "Banking and Finance", "Information Technology and Services"],
        "selectedInterests": [],
        "responses": [
            {
                "questionId": "649b3db2d50f2c4652f63bc0",
                "responseToQuestion1": "working with machines"
            },
            {
                "questionId": "649b4aa063832f79b117190e",
                "responseToQuestion2": "no"
            },
            {
                "questionId": "649b4aa063832f79b117190f",
                "responseToQuestion3": "analytical"
            },
            {
                "questionId": "649b4aa063832f79b1171910",
                "responseToQuestion4": "independently"
            },
            {
                "questionId": "649b4aa063832f79b1171911",
                "responseToQuestion5": "yes"
            },
            {
                "questionId": "649b4aa063832f79b1171912",
                "responseToQuestion6": "no"
            },
            {
                "questionId": "649b4aa063832f79b1171913",
                "responseToQuestion7": "yes"
            },
            {
                "questionId": "649b4aa063832f79b1171914",
                "responseToQuestion8": "fast paced"
            },
            {
                "questionId": "649b4aa063832f79b1171915",
                "responseToQuestion9": "no"
            },
            {
                "questionId": "649b4aa063832f79b1171916",
                "responseToQuestion10": "yes"
            },
            {
                "questionId": "649b4aa063832f79b1171917",
                "responseToQuestion11": "no"
            },
            {
                "questionId": "649b4aa063832f79b1171918",
                "responseToQuestion12": "yes"
            },
            {
                "questionId": "649b4aa063832f79b1171919",
                "responseToQuestion13": "no"
            },
            {
                "questionId": "649b4aa063832f79b117191a",
                "responseToQuestion14": "yes"
            },
            {
                "questionId": "649b4aa063832f79b117191b",
                "responseToQuestion15": "no"
            },
            {
                "questionId": "649b4aa063832f79b117191c",
                "responseToQuestion16": "no"
            },
            {
                "questionId": "649b4aa063832f79b117191d",
                "responseToQuestion17": "yes"
            },
            {
                "questionId": "649b4aa063832f79b117191e",
                "responseToQuestion18": "not important"
            },
            {
                "questionId": "649b4aa063832f79b117191f",
                "responseToQuestion19": "hybrid"
            },
            {
                "questionId": "64a86abdd8f3617c13237c93",
                "responseToQuestion20": "yes"
            }
        ]
    }

    converted_data = json_to_dataframe(input_data)

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
