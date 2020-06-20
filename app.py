import pandas as pd
import os

from surprise import Reader, Dataset, dump, KNNBasic
from flask import Flask, request, jsonify, send_from_directory

# from flask_cors import CORS
from werkzeug.exceptions import BadRequest

from markupsafe import escape
from wrapper.recommender import RecSys
from algo.IncrementalSVD import IncrementalSVD as InSVD
from algo.XQuad import re_rank
from helper.data import surprise_build_full_train_test as build_full_train_test

app = Flask(__name__)


# CORS(app)


# app.config['DEBUG'] = True

@app.route('/api/v1/users/<string:uid>', methods=['GET', 'POST'])
def get_user_recommendations(uid):
    try:
        # Escaping params.
        raw_uid = escape(uid)

        # Init Incremental SVD Model.
        recsys = RecSys('./model/insvd')

        if request.method == 'POST':
            # Get data from request.
            data = request.get_json()

            iid, rating, k = data.values()
            rating = float(rating)

            # Partial fit new rating
            recsys.model.fold_in([(raw_uid, iid, float(rating))], verbose=False)
        else:
            k = request.args.get('k', 50)

        # Get the short head and long tail items for re-ranking the recommendation list.
        short_head_items, long_tail_items = recsys.get_short_head_and_long_tail_items(threshold=20)

        # Get the recommendations then re-rank using xQuAD algorithm
        inner_uid = recsys.model.trainset.to_inner_uid(raw_uid)
        user_profile = recsys.model.trainset.ur[inner_uid]
        recommendations = re_rank(recsys.recommend(raw_uid, 1000), user_profile, short_head_items, long_tail_items,
                                  recsys.model.trainset, epochs=int(k), reg=0.1, binary=True)
        return jsonify(recommendations)
    except (ValueError, KeyError) as e:
        handle_bad_request(e)


@app.route('/api/v1/products/<string:asin>', methods=['GET'])
def get_product_neighbors(asin):
    try:
        # Init Item-based KNN model.
        recsys = RecSys('./model/iknn')
        k = request.args.get('k', 50)
        neighbors = recsys.get_k_neighbors(asin, k=int(k))

        return jsonify(neighbors)

    except (ValueError, KeyError) as e:
        handle_bad_request(e)


@app.route('/api/v1/models/insvd', methods=['GET'])
def train_svd():
    # Build dataset that fits surprise library.
    data = pd.read_csv('./data/3M_20.csv', header=0)
    train_set, test_set = build_full_train_test(data, Reader())

    # Start training.
    model = InSVD(n_factors=20, n_epochs=100, lr_all=0.005, reg_all=.1, random_state=42)
    model.fit(train_set)
    predictions = model.test(test_set)

    # Save.
    dump.dump('./model/insvd', algo=model, predictions=predictions)
    return 'Done training IncrementalSVD model.'


@app.route('/api/v1/models/iknn', methods=['GET'])
def train_iknn():
    data = pd.read_csv('./data/3M_20.csv', header=0)
    train_set, test_set = build_full_train_test(data, Reader())

    model = KNNBasic(k=50, sim_options={'name': 'cosine', 'user_based': False}, random_state=42)
    model.fit(train_set)
    predictions = model.test(test_set)

    dump.dump('./model/iknn', algo=model, predictions=predictions)
    return 'Done training Item-based KNN model.'


@app.route('/api/v1/dataset', methods=['GET', 'POST'])
def upload_dataset():
    if request.method == 'POST':
        old_data_path = './data/data-new.csv'

        if os.path.isfile(old_data_path):
            os.rename('./data/data-new.csv', old_data_path)

        data = request.get_json()

        if not data:
            return 'Empty file', 400

        df = pd.DataFrame(data['data'], columns=data['header'])
        df.to_csv('./data/data-new.csv', index=False)

        return 'Uploading success.', 200
    else:
        return send_from_directory('./data', 'data-new.csv'), 200


@app.errorhandler(BadRequest)
def handle_bad_request(e):
    return e.description, e.code


if __name__ == '__main__':
    app.run()
