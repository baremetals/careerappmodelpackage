import pickle
import numpy as np
import pandas as pd

# Load the model when the module is imported
MODEL_PATH = 'suitability_model.pkl'
model = None


def load_model():
    global model
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)


def predict(user_responses):
    if model is None:
        load_model()
    return model.predict(user_responses)


# Load the model upon importing
load_model()
