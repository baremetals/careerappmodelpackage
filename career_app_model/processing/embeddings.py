from career_app_model.config.core import config
# import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema

# Import the connection function
from career_app_model.config.milvus_server import connect_to_milvus
from career_app_model.processing.data_manager import load_dataset

# Make sure we're connected to Milvus
connect_to_milvus()

# Load the job roles data
df = load_dataset(file_name=config.embedding_config.embeddings_data_file)
# df = pd.read_csv('structured_job_roles_data.csv')

# Aggregate the skills for each job role ID
aggregated_skills = (df.groupby(config.embedding_config.embedding_group_by)[config.embedding_config.embedding_apply_to]
                     .apply(lambda x: ','.join(x)).reset_index())

# Generate embeddings using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=config.embedding_config.embedding_max_features)  # setting max features to
# control the embedding size
X = vectorizer.fit_transform(aggregated_skills[config.embedding_config.embedding_apply_to])

embeddings_aggregated = X.todense().tolist()

# Define the primary key field
role_id = FieldSchema(
    name=config.embedding_config.embedding_role_id_name,  # "role_id"
    dtype=DataType.VARCHAR,
    is_primary=config.embedding_config.embedding_is_primary,
    max_length=config.embedding_config.embedding_max_length
)

# Define the embeddings field
embedding_field = FieldSchema(
    name=config.embedding_config.embedding_field_name,
    dtype=DataType.FLOAT_VECTOR,
    dim=config.embedding_config.embedding_dimension  # Assuming the embeddings dimension is 768, modify if different.
)

# Create the collection schema
schema = CollectionSchema(
    fields=[role_id, embedding_field],
    description=config.embedding_config.embedding_collection_description
)

collection_name = config.embedding_config.embedding_collection_name  # "job_role_embeddings"

# Create the collection
collection = Collection(
    name=collection_name,
    schema=schema
)


# Function to save embeddings to Milvus
def insert_embeddings_to_milvus(embeddings_data):
    # Prepare the data for insertion
    role_ids = df[config.embedding_config.embedding_group_by].tolist()

    # Insert data into Milvus
    collection.insert([role_ids, embeddings_data])


insert_embeddings_to_milvus(embeddings_data=embeddings_aggregated)
