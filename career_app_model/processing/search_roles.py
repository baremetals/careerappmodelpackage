from pymilvus import connections, Collection
from career_app_model.config.core import config


def roles_suitability_search(industry_data, milvus_collection):
    """
    Search for job roles with an industry suitability score of 0.5 or higher.

    Parameters:
    - industry_data: A dictionary with industry names as keys and dictionaries with `industry` (suitability score)
      and `industryId` as values.
    - milvus_collection: The Milvus collection object where the job roles are stored.

    Returns:
    - A list of job roles that match the criteria.
    """

    suitable_roles = []

    for industry_name, data in industry_data.items():
        if data['industry'] >= 0.5:
            industry_id = data['industryId']
            # Construct the Milvus search expression to find roles with the matching industry ID in their metadata
            search_expression = f"metadata like '%\"industries\":%\"{industry_id}\"%'"
            # Execute the search query in the Milvus collection
            search_results = milvus_collection.search(
                data=None,  # No need to provide data for filtering based on metadata
                anns_field=config.embedding_config.embedding_field_name,
                param=dict(metric_type='L2', params=dict(nprobe=10)),  # Adjust search parameters as needed
                limit=10,  # Adjust the limit as needed
                expr=search_expression
            )
            # Process search results and extract role IDs
            for result in search_results:
                for hit in result:
                    suitable_roles.append(hit)

    return suitable_roles
