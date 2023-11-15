from pathlib import Path
from typing import List

import joblib

from career_app_model.config.core import DATASET_DIR


def create_vector(vector_list: List):
    saved_vectorizer = joblib.load(Path(f"{DATASET_DIR}/vectorizer.joblib"))

    vector_list_combined = ' '.join(vector_list)
    transform_vector_list = saved_vectorizer.transform([vector_list_combined]).todense().tolist()

    return transform_vector_list
