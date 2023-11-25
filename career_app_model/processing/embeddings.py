import ast
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema
from career_app_model.config.core import config
from career_app_model.config.milvus_server import connect_to_milvus
from career_app_model.create_and_save_vectorizer import create_and_save_vectorizer
from career_app_model.processing.data_manager import load_dataset

# Make sure we're connected to Milvus
connect_to_milvus()


def convert_string_to_list(string):
    try:
        return ast.literal_eval(string)
    except ValueError:
        return []


embeddings_aggregated = create_and_save_vectorizer()

# Load the job roles data
df = load_dataset(file_name=config.embedding_config.embeddings_data_file)
# Drop extra columns
df = df.drop(columns=['Education Requirements', 'Progress Paths', 'Skills'])

# Define the primary key field
role_id = FieldSchema(
    name="role_id",
    dtype=DataType.VARCHAR,
    is_primary=True,
    max_length=200
)

# Define the embeddings field
embedding = FieldSchema(
    name="Embedding",
    dtype=DataType.FLOAT_VECTOR,
    dim=437
)

title = FieldSchema(
    name="title",
    dtype=DataType.VARCHAR,
    max_length=255
)

required_weight = FieldSchema(name="required_weight", dtype=DataType.FLOAT)
industries = FieldSchema(name="industries", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=900,
                         max_length=1000)
career_paths = FieldSchema(name="career_paths", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=900,
                           max_length=1000)  # Serialized JSON string

description = FieldSchema(
    name="description",
    dtype=DataType.VARCHAR,
    max_length=1000
)

# Create the collection schema
schema = CollectionSchema(
    fields=[
        role_id,
        embedding,
        title,
        required_weight,
        industries,
        career_paths,
        description
    ],
    description="Job Role Embeddings"

)

collection_name = "job_role_embeddings"

# Create the collection
collection = Collection(
    name=collection_name,
    schema=schema
)

index_params = {
    "metric_type": "L2",  # Or another metric type based on your needs
    "index_type": "AUTOINDEX",  # Choose an index type suitable for your use case
    "params": {}  # Adjust parameters as needed
}
collection.create_index(field_name="Embedding", index_params=index_params)


# Function to save embeddings to Milvus
def insert_embeddings_to_milvus(embeddings_data):
    df['Embedding'] = embeddings_data
    df['industries'] = df['industries'].apply(convert_string_to_list)
    data_list = df.to_dict('records')

    # Insert data into Milvus
    collection.insert(data_list)


insert_embeddings_to_milvus(embeddings_data=embeddings_aggregated)
