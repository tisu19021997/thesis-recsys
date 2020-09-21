import pickle
from algo.XQuad import xquad
from helper.data import get_azure_ml_stream_from_blob


def save_model(file_name, model=None, predictions=None, trainset=None):
    dump_obj = {
        'model': model,
        'predictions': predictions,
        'trainset': trainset,
    }
    pickle.dump(dump_obj, open(file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def load_model_from_file(file_name):
    dump_obj = pickle.load(open(file_name, 'rb'))
    return dump_obj['model'], dump_obj['predictions'], dump_obj['trainset']


def load_model_from_blob(blob_url, model_name='insvd'):
    model_file = get_azure_ml_stream_from_blob(blob_url)
    dump_obj = pickle.loads(model_file)

    if model_name == 'insvd':
        return dump_obj['model'], dump_obj['predictions'], dump_obj['trainset']

    return dump_obj['predictions'], dump_obj['model']


def build_recommendations(recsys, uid, k=50, n=1000, short_long_threshold=3):
    # Get the short head and long tail items for re-ranking the recommendation list.
    short_head_items, long_tail_items = recsys.get_short_head_and_long_tail_items(threshold=short_long_threshold)

    # Get the base recommendations
    raw_recommendations = recsys.recommend(uid, n)

    # Get the recommendations then re-rank using xQuAD algorithm
    user_profile = recsys.get_user_profile(uid).tolist()

    # New user
    if not user_profile:
        return raw_recommendations[:k]

    # Re-rank the recommendation using xquad
    recommendations = xquad(raw_recommendations, user_profile, short_head_items, long_tail_items, n_epochs=int(k))

    return recommendations


def build_neighbors(knn_model, iid, k=50):
    inner_neighbors = knn_model.get_neighbors(knn_model.trainset.to_inner_iid(iid), k=int(k))
    raw_neighbors = [knn_model.trainset.to_raw_iid(iid) for iid in inner_neighbors]

    return raw_neighbors
