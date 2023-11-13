from typing import List

from pymilvus import Collection


def vector_search(milvus_collection: Collection, role_ids: List[dict], query_vector: List) -> List:
    # Retrieve embeddings for the industry-based results

    # Extract IDs from initial query results
    filtered_ids = [str(role["role_id"]) for role in role_ids]
    filtered_ids_str = ', '.join(filtered_ids)

    # Perform the vector search limited to the filtered IDs
    vector_search_results = milvus_collection.search(
        data=query_vector,
        anns_field="Embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=5,
        expression=f"id in [{filtered_ids_str}]"
    )
    # Return the top N most similar roles
    return vector_search_results
