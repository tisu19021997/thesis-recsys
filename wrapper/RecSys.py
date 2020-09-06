import pandas as pd
from helper.modeling import load_model
from collections import defaultdict


class RecSys:
    def __init__(self, model_path):
        self.model, self.predictions, self.trainset = load_model(model_path)

    def recommend(self, u_id, n=50):
        # Get all products
        all_items = self.get_full_dataset()['i_id'].unique()
        user_profile = set(self.get_user_profile(u_id))

        # Predict the user rating on each product
        recommendations = []
        for i_id in all_items:
            if i_id in user_profile:
                continue

            prediction = self.model.predict_pair(u_id, i_id)
            recommendations.append((i_id, prediction))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n]

    def get_full_dataset(self):
        return pd.concat([self.trainset, self.predictions.drop(columns=['prediction'])])

    def get_testset(self):
        return self.predictions.drop(columns=['prediction'])

    def get_top_n(self, n=10, from_test_set=False):
        """Returns a dictionary of top-N recommendations if `from_test_set` is False,
        otherwise top-N highest rated products, for each user in test set.
        Args:
            n (int): Number of maximum products for each user recommendation.
            from_test_set (bool): Whether to get the top-N highest rated products.
        """
        # Get top n recommendation from test set
        top_n = defaultdict(list)

        for uid, iid, true_r, est in self.predictions.values:
            score = true_r if from_test_set else est
            top_n[uid].append((iid, score))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n

    def get_short_head_and_long_tail_items(self, threshold):
        """Returns the short head and long tail items as a tuple.
        Args:
            threshold (int): If user's number of ratings larger than the threshold, it is in short-head.
              Else, it is in long-tail.
        """
        # using set for faster look-up time
        dataset = self.get_full_dataset()
        data_items = dataset.groupby(by='i_id').count().reset_index()

        short_head_items = set(data_items[data_items['rating'] >= threshold]['i_id'].values)
        long_tail_items = set(data_items[data_items['rating'] < threshold]['i_id'].values)

        self.short_head = short_head_items
        self.long_tail = long_tail_items

        return short_head_items, long_tail_items

    def get_user_profile(self, u_id):
        """Return the user's rated products as a numpy.ndarray
        Args:
            u_id (str): User's id
        """
        return self.trainset[self.trainset['u_id'] == u_id]['i_id'].unique()
