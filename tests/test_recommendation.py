from typing import List

from career_app_model.recommendation import recommend_roles


def test_make_recommendation(user_interest_data):
    recommendation_results = recommend_roles(user_interest_data)
    print("The Returned Results", recommendation_results)
    assert isinstance(recommendation_results, List)


def test_make_recommendation_with_vectors(user_interest_data):
    user_interests = ["analytical thinking skills", "excellent written communication skills"]
    recommendation_results = recommend_roles(user_interest_data, user_interests)
    print("The Returned Results", recommendation_results)
    assert isinstance(recommendation_results, List)
