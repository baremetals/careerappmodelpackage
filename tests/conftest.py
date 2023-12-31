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
        "profileId": "65606f82707d9d2d67b59a8e",
        "selectedIndustries": [
            "Banking and Finance",
            "Professional Services",
            "Information Technology and Services"
        ],
        "selectedInterests": [
            "analytical thinking skills",
            "excellent written communication skills"
        ],
        "responses": [
            {
                "questionId": "64d2bcef1902d8ba6a46c0e8",
                "questionVersion": 1,
                "questionNumber": 1,
                "responseId": "64d2bcf01902d8ba6a46c17e",
                "responseToQuestion": "ResponseOption3"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0e9",
                "questionVersion": 1,
                "questionNumber": 2,
                "responseId": "64d2bcf01902d8ba6a46c180",
                "responseToQuestion": "ResponseOption1"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0ea",
                "questionVersion": 1,
                "questionNumber": 3,
                "responseId": "64d2bcf01902d8ba6a46c18d",
                "responseToQuestion": "ResponseOption2"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0eb",
                "questionVersion": 1,
                "questionNumber": 4,
                "responseId": "64d2bcf01902d8ba6a46c190",
                "responseToQuestion": "ResponseOption1"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0ec",
                "questionVersion": 1,
                "questionNumber": 5,
                "responseId": "64d2bcf01902d8ba6a46c184",
                "responseToQuestion": "ResponseOption1"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0ed",
                "questionVersion": 1,
                "questionNumber": 6,
                "responseId": "64d2bcf01902d8ba6a46c195",
                "responseToQuestion": "ResponseOption2"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0ee",
                "questionVersion": 1,
                "questionNumber": 7,
                "responseId": "64d2bcf01902d8ba6a46c189",
                "responseToQuestion": "ResponseOption2"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0ef",
                "questionVersion": 1,
                "questionNumber": 8,
                "responseId": "64d2bcf01902d8ba6a46c1a5",
                "responseToQuestion": "ResponseOption2"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0f0",
                "questionVersion": 1,
                "questionNumber": 9,
                "responseId": "64d2bcf11902d8ba6a46c1d6",
                "responseToQuestion": "ResponseOption3"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0f1",
                "questionVersion": 1,
                "questionNumber": 10,
                "responseId": "64d2bcf01902d8ba6a46c1aa",
                "responseToQuestion": "ResponseOption3"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0f2",
                "questionVersion": 1,
                "questionNumber": 11,
                "responseId": "64d2bcf01902d8ba6a46c1ac",
                "responseToQuestion": "ResponseOption1"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0f3",
                "questionVersion": 1,
                "questionNumber": 12,
                "responseId": "64d2bcf01902d8ba6a46c1ce",
                "responseToQuestion": "ResponseOption3"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0f4",
                "questionVersion": 1,
                "questionNumber": 13,
                "responseId": "64d2bcf01902d8ba6a46c1d2",
                "responseToQuestion": "ResponseOption3"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0f5",
                "questionVersion": 1,
                "questionNumber": 14,
                "responseId": "64d2bcf01902d8ba6a46c1b0",
                "responseToQuestion": "ResponseOption1"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0f6",
                "questionVersion": 1,
                "questionNumber": 15,
                "responseId": "64d2bcf01902d8ba6a46c1b5",
                "responseToQuestion": "ResponseOption2"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0f7",
                "questionVersion": 1,
                "questionNumber": 16,
                "responseId": "64d2bcf01902d8ba6a46c1b8",
                "responseToQuestion": "ResponseOption1"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0f8",
                "questionVersion": 1,
                "questionNumber": 17,
                "responseId": "64d2bcf01902d8ba6a46c1bd",
                "responseToQuestion": "ResponseOption2"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0f9",
                "questionVersion": 1,
                "questionNumber": 18,
                "responseId": "64d2bcf01902d8ba6a46c1c1",
                "responseToQuestion": "ResponseOption2"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0fa",
                "questionVersion": 1,
                "questionNumber": 19,
                "responseId": "64d2bcf01902d8ba6a46c1c4",
                "responseToQuestion": "ResponseOption1"
            },
            {
                "questionId": "64d2bcef1902d8ba6a46c0fb",
                "questionVersion": 1,
                "questionNumber": 20,
                "responseId": "64d2bcf01902d8ba6a46c1ca",
                "responseToQuestion": "ResponseOption3"
            }
        ]
    }

    return input_data
