from surprise import Dataset
from surprise.model_selection import train_test_split
from flask import jsonify


def surprise_build_train_test(data_frame, reader, full=True):
    """Returns tuple of `surprise` full train set and test set.
    Args:
        data_frame: Pandas data-frame.
        reader: `surprise` `Reader` instance.
        full: Whether to build full trainset or split one
    """
    raw_data = Dataset.load_from_df(data_frame, reader=reader)

    if full:
        train_set = raw_data.build_full_trainset()
        test_set = train_set.build_testset()

        return train_set, test_set

    return train_test_split(raw_data, test_size=0.2)


def get_products_from_ratings(ratings, tolist=False):
    return ratings.asin if not tolist else ratings.asin.values.tolist()


def get_reviewers_from_ratings(ratings, tolist=False):
    return ratings.reviewerID if not tolist else ratings.reviewerID.values.tolist()


def is_header_valid(data_header):
    # Check header.
    if data_header != ['reviewerID', 'asin', 'overall']:
        return False

    return True
