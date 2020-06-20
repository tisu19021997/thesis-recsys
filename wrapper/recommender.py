from surprise import dump
from surprise.accuracy import rmse, mae
from helper.accuracy import precision_recall_at_k


class RecSys:
    """A wrapper class to load pre-trained `surprise` model and making predictions.
    Attributes:
         predictions (tuple): Tuple of (user_id, item_id, true_rating, estimated_rating, details)
         model: Pre-trained model
    """

    def __init__(self, model_path):
        """Init RecSys
        Args:
            model_path (str): Model path
        """
        # load prediction and model from a given file
        self.predictions, self.model = dump.load(model_path)

        self.avg_recall = 0
        self.avg_precision = 0

    def recommend(self, raw_uid, k=50):
        """Return top-K recommendations for a user.

        Args:
            raw_uid (str): Raw user id then it will be converted to inner id of the model.
            k (int): Number of recommendations.
        """
        # exclude items that user has rated
        inner_uid = self.model.trainset.to_inner_uid(raw_uid)
        user_profile = self.model.trainset.ur[inner_uid]
        rated_items = set([inner_iid for inner_iid, _ in user_profile])

        recommendations = []

        # looping through each item in train set, predict the rating of each pair of item and user
        for iid in self.model.trainset.all_items():
            if iid in rated_items:
                continue

            rid = self.model.trainset.to_raw_iid(iid)
            prediction = self.model.predict(raw_uid, rid)

            # only consider rating that are larger or equal than 3
            if prediction.est >= 3:
                recommendations.append((rid, prediction.est))

        # sort by rating
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:k]

    def get_short_head_and_long_tail_items(self, threshold):
        """Returns the short head and long tail items as a tuple.
        Args:
            threshold (int): If user's number of ratings larger than the threshold, it is in short-head.
             Else, it is in long-tail.
        """
        # using set for faster look-up time
        short_head_items = set()
        long_tail_items = set()

        for inner_iid, ratings in self.model.trainset.ir.items():
            raw_iid = self.model.trainset.to_raw_iid(inner_iid)
            if len(ratings) >= threshold:
                short_head_items.add(raw_iid)
            else:
                long_tail_items.add(raw_iid)

        return short_head_items, long_tail_items

    def get_k_neighbors(self, raw_id, user_based=False, k=50):
        """Returns Top-K neighbors of a user of item (only for KNN-inspired models).
        Args:
            user_based (bool): User or item based neighbors
            raw_id (str): User/item raw id
            k (int): Number of neighbors
        """
        if user_based:
            to_inner = self.model.trainset.to_inner_uid
            to_raw = self.model.trainset.to_raw_uid
        else:
            to_inner = self.model.trainset.to_inner_iid
            to_raw = self.model.trainset.to_raw_iid

        inner_id = to_inner(raw_id)
        inner_neighbors = self.model.get_neighbors(inner_id, k=k)

        raw_neighbors = [to_raw(iid) for iid in inner_neighbors]

        return raw_neighbors

    def compute_precision_recall_at_k(self, k=20, threshold=3.0):
        """Return precision and recall @K of current model
        Args:
            k (int): Number of K to stop.
            threshold (float): Minimum rating for a product to be considered as "relevant"
        """
        precisions, recalls = precision_recall_at_k(self.predictions, k=k, threshold=threshold)

        self.avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
        self.avg_recall = sum(rec for rec in recalls.values()) / len(recalls)

        return self.avg_precision, self.avg_recall

    def compute_f1(self):
        """Returns F1 score of current model"""
        return 2 * (self.avg_precision * self.avg_recall) / (self.avg_precision + self.avg_recall)

    def compute_rmse(self):
        """Returns Root-mean-square error of current model"""
        return rmse(self.predictions, verbose=False)

    def compute_mae(self):
        """Returns Mean absolute error of current model"""
        return mae(self.predictions, verbose=False)
