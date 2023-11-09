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
