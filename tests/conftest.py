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
                "questionId": "64d2bcef1902d8ba6a46c0e8",
                "responseToQuestion1": "ResponseOption3"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0e9",
                "responseToQuestion2": "ResponseOption1"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0ea",
                "responseToQuestion3": "ResponseOption2"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0ee",
                "responseToQuestion4": "ResponseOption2"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0eb",
                "responseToQuestion5": "ResponseOption1"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0ec",
                "responseToQuestion6": "ResponseOption1"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0ed",
                "responseToQuestion7": "ResponseOption2"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0f0",
                "responseToQuestion8": "ResponseOption3"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0ef",
                "responseToQuestion9": "ResponseOption2"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0f1",
                "responseToQuestion10": "ResponseOption3"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0f2",
                "responseToQuestion11": "ResponseOption1"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0f5",
                "responseToQuestion12": "ResponseOption1"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0f6",
                "responseToQuestion13": "ResponseOption2"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0f7",
                "responseToQuestion14": "ResponseOption1"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0f8",
                "responseToQuestion15": "ResponseOption2"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0f9",
                "responseToQuestion16": "ResponseOption2"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0fa",
                "responseToQuestion17": "ResponseOption1"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0fb",
                "responseToQuestion18": "ResponseOption3"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0f3",
                "responseToQuestion19": "ResponseOption3"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0f4",
                "responseToQuestion20": "ResponseOption3"
            }
        ]
    }

    return input_data
