from career_app_model.processing.embeddings import create_vectorizer
from career_app_model.processing.search_roles import roles_suitability_search
from prepare_data import generate_query_embedding
from pymilvus import connections, Collection
from filter import filter_results, filter_by_interest_embedding
from career_app_model.config.core import config
from career_app_model.config.milvus_server import connect_to_milvus

# Make sure we're connected to Milvus
connect_to_milvus()


def search_suitable_roles(industry_data, user_interests, top_n=5):
    milvus_collection = Collection(config.embedding_config.embedding_collection_name)
    # Step 1: Industry-based search
    industry_based_results = roles_suitability_search(industry_data, milvus_collection)

    vectorizer = create_vectorizer()

    # Step 2: Interest-based search, if interests are provided
    if user_interests:
        # Generate the query embedding for user interests
        query_embedding = vectorizer.transform([' '.join(user_interests)]).todense()

        # Filter the industry-based results using the interest embedding
        interest_based_results = filter_by_interest_embedding(query_embedding, industry_based_results,
                                                              milvus_collection, top_n)

        # If we found interest-based results, return them
        if interest_based_results:
            return interest_based_results

    # Step 3: Return the industry-based results if no interests are provided or no interest-based results were found
    return industry_based_results

# def search_job_roles(user_interests, suitability_scores):
#     # Generate a query embedding from user interests
#     query_embedding = generate_query_embedding(user_interests)
#
#     # Connect to Milvus
#     connections.connect("default")
#     collection = Collection("job_role_embeddings")
#
#     # Perform the search in Milvus
#     search_params = {
#         "metric_type": "IP",  # Inner Product or other similarity metric
#         "params": {"nprobe": 10},
#     }
#     search_results = collection.search(
#         data=[query_embedding],
#         anns_field="embedding",
#         param=search_params,
#         limit=10,
#         expr=None
#     )
#
#     # Filter results based on suitability scores and return them
#     filtered_results = filter_results(search_results, suitability_scores)
#
#     return filtered_results
