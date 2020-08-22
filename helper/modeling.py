import pickle
from algo.XQuad import xquad


def save_model(file_name, model=None, predictions=None, trainset=None):
    dump_obj = {
        'model': model,
        'predictions': predictions,
        'trainset': trainset,
    }
    pickle.dump(dump_obj, open(file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def load_model(file_name):
    dump_obj = pickle.load(open(file_name, 'rb'))
    return dump_obj['model'], dump_obj['predictions'], dump_obj['trainset']


def build_recommendations(recsys, uid, k=50, n=1000, short_long_threshold=3):
    # Get the short head and long tail items for re-ranking the recommendation list.
    short_head_items, long_tail_items = recsys.get_short_head_and_long_tail_items(threshold=short_long_threshold)

    # Get the recommendations then re-rank using xQuAD algorithm
    user_profile = recsys.get_user_profile(uid)
    # Get the base recommendations
    raw_recommendations = recsys.recommend(uid, n)
    # Re-rank the recommendation using xquad
    recommendations = xquad(raw_recommendations, user_profile, short_head_items, long_tail_items, n_epochs=int(k))

    return recommendations


def build_neighbors(knn_model, iid, k=50):
    inner_neighbors = knn_model.get_neighbors(knn_model.trainset.to_inner_iid(iid), k=int(k))
    raw_neighbors = [knn_model.trainset.to_raw_iid(iid) for iid in inner_neighbors]

    return raw_neighbors
