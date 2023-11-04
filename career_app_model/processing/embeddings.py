from config.core import config
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema

# Import the connection function
from career_app_model.config.milvus_server import connect_to_milvus

# Make sure we're connected to Milvus
connect_to_milvus()

# Load the job roles data
df = pd.read_csv('structured_jobroles.csv')

# Aggregate the skills for each job role ID
aggregated_skills = df.groupby('Role ID')['Skills'].apply(lambda x: ','.join(x)).reset_index()

# Generate embeddings using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # setting max features to control the embedding size
X = vectorizer.fit_transform(aggregated_skills['Skills'])

embeddings_aggregated = X.todense().tolist()

# Define the primary key field
role_id = FieldSchema(
    name="role_id",
    dtype=DataType.VARCHAR,
    is_primary=True,
    max_length=200
)

# Define the embeddings field
embedding_field = FieldSchema(
    name="embedding",
    dtype=DataType.FLOAT_VECTOR,
    dim=437  # Assuming the embeddings dimension is 768, modify if different.
)

# Create the collection schema
schema = CollectionSchema(
    fields=[role_id, embedding_field],
    description="Job Role Embeddings"
)

collection_name = "job_role_embeddings"

# Create the collection
collection = Collection(
    name=collection_name,
    schema=schema
)


# Function to save embeddings to Milvus
def insert_embeddings_to_milvus(embeddings_data):
    # Prepare the data for insertion
    role_ids = df['Role ID'].tolist()

    # Insert data into Milvus
    collection.insert([role_ids, embeddings_data])


insert_embeddings_to_milvus(embeddings_data=embeddings_aggregated)
