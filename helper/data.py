from surprise import Dataset


def surprise_build_full_train_test(data_frame, reader):
    """Returns tuple of `surprise` full train set and test set.
    Args:
        data_frame: Pandas data-frame.
        reader: `surprise` `Reader` instance.
    """
    raw_data = Dataset.load_from_df(data_frame, reader=reader)
    train_set = raw_data.build_full_trainset()
    test_set = train_set.build_testset()

    return train_set, test_set


def get_products_from_ratings(ratings, tolist=False):
    return ratings.asin if not tolist else ratings.asin.values.tolist()


def get_reviewers_from_ratings(ratings, tolist=False):
    return ratings.reviewerID if not tolist else ratings.reviewerID.values.tolist()
