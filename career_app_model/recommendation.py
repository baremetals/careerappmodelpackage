from typing import List, Optional

from career_app_model.processing.search_roles import roles_suitability_search
from career_app_model.processing.vector_search import vector_search
from pymilvus import Collection

from career_app_model.config.core import config
from career_app_model.config.milvus_server import connect_to_milvus
from career_app_model.processing.create_vector import create_vector

# Make sure we're connected to Milvus
connect_to_milvus()


def recommend_roles(industry_data: List, user_interests: Optional[List[str]] = None) -> List:
    collection = Collection(config.embedding_config.embedding_collection_name)
    # Step 1: Industry-based search
    roles_search_results = roles_suitability_search(industry_data, collection)

    # Step 2: Interest-based search, if interests are provided
    if user_interests:
        # Generate the query embedding for user interests
        query_embedding = create_vector(user_interests)

        # Filter the industry-based results using the interest embedding
        vector_search_results = vector_search(collection, roles_search_results, query_embedding)

        # If we found interest-based results, return them
        if len(vector_search_results) > 0:
            return vector_search_results

    # Step 3: Return the industry-based results if no interests are provided or no interest-based results were found
    return roles_search_results
