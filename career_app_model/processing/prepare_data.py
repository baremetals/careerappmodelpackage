# from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple
from career_app_model.datasets.industry_names import get_industry_names
import numpy as np
import pandas as pd


# from embeddings import create_vectorizer


# Function to prepare data for training
def prepare_data(user_data_df: pd.DataFrame, structured_data_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    industry_names = get_industry_names()

    for _, user_row in user_data_df.iterrows():
        features = []
        for q_idx in range(1, 21):
            response_idx = int(user_row[f'responseToQuestion{q_idx}'].split('ResponseOption')[-1])
            response_weights = structured_data_df.iloc[q_idx - 1, 8 + 3 * (response_idx - 1)::9].values
            features.extend(response_weights)

        # Calculate the score for each industry
        industry_scores = {}
        for industry in industry_names:
            industry_score = 0
            for q_idx in range(1, 21):
                if f'responseToQuestion{q_idx}' in user_row:
                    response_option = user_row[f'responseToQuestion{q_idx}']
                    option_idx = int(response_option.split('ResponseOption')[-1]) - 1
                    start_idx = structured_data_df.columns.get_loc(f"{industry}_Option1")
                    end_idx = start_idx + 3

                    weights = [0] * 3
                    weights[option_idx] = 1

                    question_weights = structured_data_df.iloc[q_idx - 1, start_idx:end_idx].values
                    industry_score += np.dot(question_weights, weights)

            industry_scores[industry] = industry_score

            # Convert the scores dictionary to a list in the order of industry_names
        y.append([industry_scores[industry] for industry in industry_names])
        X.append(features)

    return np.array(X), np.array(y)


def transform_input_for_prediction(input_data: dict, structured_data_df: pd.DataFrame):
    user_features = []

    for response_item in input_data["responses"]:
        q_number = response_item["questionNumber"]
        response = response_item["responseToQuestion"]
        response_idx = int(response.split('ResponseOption')[-1])
        print(f"Question Number: {q_number}, Response Index: {response_idx}")

        response_weights = structured_data_df.iloc[q_number - 1, 8 + 3 * (response_idx - 1)::9].values
        print(f"Response Weights: {response_weights}")

        user_features.extend(response_weights)

    return np.array([user_features])


def json_to_dataframe(input_data: dict) -> pd.DataFrame:
    """Converts the provided JSON structure into a pandas DataFrame in the expected format."""

    # Extracting the main fields
    user_id = input_data["userId"]
    industries = input_data.get("selectedIndustries", [])

    selected_industries = {
        "selectedIndustry1": industries[0] if len(industries) > 0 else None,
        "selectedIndustry2": industries[1] if len(industries) > 1 else None,
        "selectedIndustry3": industries[2] if len(industries) > 2 else None,
        "selectedIndustry4": industries[3] if len(industries) > 3 else None,
        "selectedIndustry5": industries[4] if len(industries) > 4 else None,
    }

    # Extracting the responses
    responses = {}
    for resp in input_data["responses"]:
        responses.update(resp)
        # Remove the 'questionId' key as it's not needed in the final DataFrame
        responses.pop("questionId", None)

    # Combining all the extracted data
    data = {
        "userId": user_id,
        **selected_industries,
        **responses
    }

    # Convert to DataFrame
    df = pd.DataFrame([data])

    return df

# def generate_query_embedding(user_interests):
#     vectorizer = create_vectorizer()
#     interests_str = ",".join(user_interests)
#
#     # Generate an embedding for the given interests
#     interests_embedding = vectorizer.fit_transform([interests_str]).todense().tolist()
#     return interests_embedding
