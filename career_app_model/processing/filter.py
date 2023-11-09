import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# def filter_results(search_results, suitability_scores):
#     # Process the search results
#     suitable_job_roles = []
#     for result in search_results:
#         for job in result:
#             # Load the metadata for each job
#             metadata = json.loads(job.entity.get("metadata"))
#
#             # Check if the job's industry is within the user's suitability scores
#             if any(industry in suitability_scores and suitability_scores[industry] >= 0.5 for industry in
#                    metadata['industries']):
#                 suitable_job_roles.append(job)
#
#     return suitable_job_roles


def filter_by_interest_embedding(query_embedding, industry_based_results, milvus_collection, top_n):
    # Retrieve embeddings for the industry-based results
    embeddings = [role.embedding for role in
                  industry_based_results]  # Adjust this based on how you can retrieve embeddings

    # Calculate similarity scores
    similarities = cosine_similarity(query_embedding, embeddings)

    # Sort the roles by similarity score in descending order and take top N
    sorted_indices = np.argsort(similarities[0])[::-1][:top_n]

    # Return the top N most similar roles
    return [industry_based_results[index] for index in sorted_indices]
