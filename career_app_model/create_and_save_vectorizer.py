from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from career_app_model.config.core import DATASET_DIR, config
from career_app_model.processing.data_manager import load_dataset


def create_vectorizer():
    return TfidfVectorizer(max_features=config.embedding_config.embedding_max_features)


def create_and_save_vectorizer():
    # Load the job roles data
    df = load_dataset(file_name=config.embedding_config.embeddings_data_file)

    # Aggregate the skills for each job role ID
    aggregated_skills = (
        df.groupby(config.embedding_config.embedding_group_by)[config.embedding_config.embedding_apply_to]
        .apply(lambda x: ','.join(x)).reset_index())

    # Generate embeddings using TfidfVectorizer
    vectorizer = create_vectorizer()
    vectorizer.fit(aggregated_skills[config.embedding_config.embedding_apply_to])

    # save the vectorizer
    joblib.dump(vectorizer, Path(f"{DATASET_DIR}/vectorizer.joblib"))

    # control the embedding size
    X_embeddings = vectorizer.transform(aggregated_skills[config.embedding_config.embedding_apply_to])

    embeddings_aggregated = X_embeddings.todense().tolist()

    return embeddings_aggregated


if __name__ == "__main__":
    create_and_save_vectorizer()
