def get_products_from_ratings(ratings, tolist=False):
    return ratings.asin if not tolist else ratings.asin.values.tolist()


def get_reviewers_from_ratings(ratings, tolist=False):
    return ratings.reviewerID if not tolist else ratings.reviewerID.values.tolist()