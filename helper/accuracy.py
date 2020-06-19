from collections import defaultdict


def precision_recall_at_k(predictions: tuple, k: int = 10, threshold: float = 3.0) -> tuple:
    """ Return a tuple of precision and recall @K for a given predictions.
    Args:
        predictions (tuple): Tuple of predictions
            (user_id, item_id, true_rating, estimated_rating, details).
        k (int):
        threshold (float): Minimum rating for a product to be considered as "relevant".
    """

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls


def arp(recommend_list: defaultdict, trainset, testset: [tuple]):
    """Returns the Average Recommendation Popularity of a recommendation list.
    Args:
        recommend_list (defaultdict): All users' recommendation list generated.
         by the base algorithm [(raw_iid, rating)].
        trainset: `surprise.trainset`
        testset (list of tuple): List of ratings [(raw_uid, raw_iid, rating)]
    """
    num_users = len(set(raw_uid for raw_uid, _, _ in testset))
    popularity = 0

    for _, recommends in recommend_list.items():
        num_recommends = len(recommends)
        total_rates = 0
        for raw_iid, _ in recommends:
            try:
                times_rated = len(trainset.ir[trainset.to_inner_iid(raw_iid)])
            except (ValueError, KeyError):
                times_rated = 0
            total_rates += times_rated
        popularity += (total_rates / num_recommends)

    return popularity / num_users


def aplt(recommend_list: defaultdict, testset: [tuple], long_tail_items: set):
    """Returns the Average Percentage of Long Tail Items.
    Args:
        recommend_list (defaultdict): All users' recommendation list generated.
        testset (list of tuple): List of ratings [(raw_uid, raw_iid, rating)]
        long_tail_items (set): Set of items in long-tail.
    """
    num_users = len(set(raw_uid for raw_uid, _, _ in testset))

    long_tail_ratio = 0

    # l_u is user recommended list
    for _, l_u in recommend_list.items():
        long_tail_ratio += sum((i in long_tail_items for i, _ in l_u)) / len(l_u)

    return long_tail_ratio / num_users


def aclt(recommend_list: defaultdict, long_tail_items: set):
    """ Returns the Average Coverage of Long Tail items.
    Args:
        recommend_list (defaultdict): All users' recommendation list generated.
        long_tail_items (set): Set of items in long-tail.
    """
    items_set = set()
    count = 0
    for _, l_u in recommend_list.items():
        for raw_iid, _ in l_u:
            if raw_iid in items_set:
                continue

            if raw_iid in long_tail_items:
                items_set.add(raw_iid)
                count += 1

    return count / len(long_tail_items)
