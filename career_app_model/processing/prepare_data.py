import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple
from embeddings import create_vectorizer


# Function to prepare data for training
def prepare_data(user_data_df: pd.DataFrame, structured_data_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []

    for _, user_row in user_data_df.iterrows():
        features = []
        for q_idx in range(1, 21):
            response_idx = int(user_row[f'responseToQuestion{q_idx}'].split('ResponseOption')[-1])
            response_weights = structured_data_df.iloc[q_idx - 1, 8 + 3 * (response_idx - 1)::9].values
            features.extend(response_weights)

        X.append(features)

        # Calculate the average suitability score for the user's selected industries
        scores = []
        for i_idx in range(1, 6):
            industry = user_row[f'selectedIndustry{i_idx}']
            if pd.notna(industry) and industry in structured_data_df.columns:
                start_idx = structured_data_df.columns.get_loc(f"{industry}_Option1")
                end_idx = start_idx + 3
                weights = structured_data_df.iloc[20 - 1, start_idx:end_idx]
                score = np.dot(weights,
                               [1 if user_row[f'responseToQuestion{20}'] == f'ResponseOption{j + 1}' else 0 for j in
                                range(3)])
                scores.append(score)
            else:
                scores.append(0)

        y.append(np.mean(scores) if scores else 0)

    return np.array(X), np.array(y)


def json_to_dataframe(input_data: dict) -> pd.DataFrame:
    """Converts the provided JSON structure into a pandas DataFrame in the expected format."""

    # Extracting the main fields
    user_id = input_data["userId"]
    selected_industries = {
        "selectedIndustry1": input_data.get("selectedIndustries")[0] if len(
            input_data.get("selectedIndustries", [])) > 0 else None,
        "selectedIndustry2": input_data.get("selectedIndustries")[1] if len(
            input_data.get("selectedIndustries", [])) > 1 else None,
        "selectedIndustry3": input_data.get("selectedIndustries")[2] if len(
            input_data.get("selectedIndustries", [])) > 2 else None,
        "selectedIndustry4": input_data.get("selectedIndustries")[3] if len(
            input_data.get("selectedIndustries", [])) > 3 else None,
        "selectedIndustry5": input_data.get("selectedIndustries")[4] if len(
            input_data.get("selectedIndustries", [])) > 4 else None,
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


def generate_query_embedding(user_interests):
    vectorizer = create_vectorizer()
    interests_str = ",".join(user_interests)

    # Generate an embedding for the given interests
    interests_embedding = vectorizer.transform([interests_str]).todense()
    return interests_embedding
