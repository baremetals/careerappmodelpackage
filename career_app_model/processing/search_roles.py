from typing import List

from pymilvus import Collection


def roles_suitability_search(industry_data: List, milvus_collection: Collection) -> List:
    milvus_collection.load()

    suitable_roles = []

    for data in industry_data:
        if data['score'] >= 0.5:
            industry_id = data['industryId']
            industry_score = data['score']

            # Construct the Milvus search expression to find roles with the matching industry ID in their metadata
            search_expression = f"array_contains(industries, '{industry_id}') && required_weight <= {industry_score}"

            # Execute the search query in the Milvus collection
            search_results: List[dict] = milvus_collection.query(
                expr=search_expression,
                limit=10,
                output_fields=['role_id', 'industries']
            )
            # Process search results and extract role IDs
            for result in search_results:
                suitable_roles.append(result)

    return suitable_roles
