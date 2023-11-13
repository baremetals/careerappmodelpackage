import pytest

from career_app_model.config.core import config
from career_app_model.processing.data_manager import load_dataset


@pytest.fixture()
def user_sample_input_data():
    return load_dataset(file_name=config.app_config.test_data_file)


def model_sample_input_data():
    return load_dataset(file_name=config.app_config.training_data_file)


def embedding_sample_input_data():
    return load_dataset(file_name=config.embedding_config.embedding_data_file)


@pytest.fixture()
def user_interest_data():
    data_list = [
        {
            "industryId": "649b600bfe840a75193787e7",
            "industryName": "Professional Services",
            "score": 0.74
        },
        {
            "industryId": "649b600bfe840a75193787d4",
            "industryName": "Banking and Finance",
            "score": 0.5
        },
        {
            "industryId": "649b600bfe840a75193787d7",
            "industryName": "Construction and Engineering",
            "score": 0.612
        },
        {
            "industryId": "649b5f0afe840a75193787cc",
            "industryName": "Aerospace",
            "score": 0.44
        },
        {
            "industryId": "649b600bfe840a75193787db",
            "industryName": "Entertainment and Media",
            "score": 0.24
        }
    ]

    return data_list


@pytest.fixture()
def test_sample_data():
    input_data = {
        "userId": "64b16ff9746c9b729c5",
        "selectedIndustries": ["Banking and Finance", "Banking and Finance", "Information Technology and Services"],
        "selectedInterests": ["analytical thinking skills", "excellent written communication skills"],
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

    return input_data
