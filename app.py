import requests
import json
import pandas as pd
import os
from zipfile import ZipFile

from surprise import Reader, dump, KNNBasic
from surprise.accuracy import rmse, mae
from flask import Flask, request, Response, jsonify, send_from_directory, after_this_request

from flask_cors import CORS
from werkzeug.exceptions import BadRequest, abort
from markupsafe import escape

from wrapper.recommender import RecSys
from algo.IncrementalSVD import IncrementalSVD as InSVD
from algo.XQuad import re_rank
from helper.data import surprise_build_train_test as build_train_test
from auth.auth import BearerAuth

app = Flask(__name__)
CORS(app, expose_headers='X-Model-Info')


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


@app.route('/api/v1/models', methods=['POST'])
def train_model():
    # Send request to Nodejs server for authentication.
    _, jwt = request.headers['Authorization'].split()
    r = requests.get('http://127.0.0.1:8081/api/v1/auth/admin', auth=BearerAuth(jwt))

    if r.status_code != 200:
        return abort(r.status_code)

    # Extract data from request.
    data = request.get_json()
    dataset, data_header, model_name, params, train_type, save_on_server, save_on_local = data.values()

    # Use the data uploaded or data on server.
    df = pd.DataFrame(dataset, columns=data_header) if dataset else pd.read_csv('./data/data-new.csv', header=0)
    train_set, test_set = build_train_test(df, Reader(), full=train_type == 'full')

    if model_name == 'insvd':
        n_factors, n_epochs, lr_all, reg_all, random_state = params.values()

        # Parse data types.
        n_factors = int(n_factors)
        n_epochs = int(n_epochs)
        lr_all = float(lr_all)
        reg_all = float(reg_all)
        random_state = int(random_state)

        model = InSVD(n_factors=n_factors, n_epochs=n_epochs,
                      lr_all=lr_all, reg_all=reg_all, random_state=random_state)
    else:
        k, sim_options, random_state = params.values()

        model = KNNBasic(k=k, sim_options={'name': sim_options, 'user_based': False}, random_state=random_state)

    # Fitting and testing.
    model.fit(train_set)
    predictions = model.test(test_set)

    # Add suffix if not save on server.
    model_path = f'./model/{model_name}' if save_on_server else f'./model/{model_name}-temp'

    # Save.
    dump.dump(model_path, algo=model, predictions=predictions)
    model_info = {
        'rmse': rmse(predictions),
        'mae': mae(predictions),
    }

    # Zip the trained model.
    try:
        zip_obj = ZipFile(f'{model_path}.zip', 'w')
        zip_obj.write(model_path)
        zip_obj.close()
    except FileNotFoundError:
        return abort(404)

    @after_this_request
    def remove_dump_files(response):
        # If not save model on server, delete model dump file.
        if not save_on_server:
            os.remove(model_path)

        # Always delete the .zip file.
        os.remove(f'{model_path}.zip')

        return response

    if save_on_local:
        with open(f'{model_path}.zip', 'rb') as f:
            model_zip = f.readlines()

        resp = Response(model_zip)
        resp.headers['X-Model-Info'] = json.dumps(model_info)
        resp.headers['Content-Type'] = 'application/zip'
        resp.headers['Content-Disposition'] = 'attachment; filename=%s;' % 'model.zip'

        return resp
        # return Response(model_zip, headers={
        #     'X-Info': json.dumps(model_info),
        #     'Content-Type': 'application/zip',
        #     'Content-Disposition': 'attachment; filename=%s;' % 'model.zip',
        # })

        # return send_from_directory('./model', model_file), 200

    return jsonify(model_info)


@app.route('/api/v1/models/test', methods=['GET'])
def test_model():
    preds, model = dump.load('./model/insvd')
    rmse(preds)

    return 'OK', 200


@app.route('/api/v1/dataset', methods=['GET', 'POST'])
def upload_dataset():
    # Send request to Nodejs server for authentication.
    _, jwt = request.headers['Authorization'].split()
    r = requests.get('http://127.0.0.1:8081/api/v1/auth/admin', auth=BearerAuth(jwt))

    if r.status_code != 200:
        return abort(r.status_code)

    # Posting new dataset.
    if request.method == 'POST':
        old_data_path = './data/data-old.csv'
        new_data_path = './data/data-new.csv'

        # If the old dataset exists, rename it to *-old.csv
        if os.path.isfile(new_data_path):
            os.rename(new_data_path, old_data_path)

        data = request.get_json()

        if not data['data']:
            return jsonify({'message': 'Empty data.'})

        # Save as a csv file.
        df = pd.DataFrame(data['data'], columns=data['header'])
        df.to_csv(new_data_path, index=False)

        return jsonify({'message': 'Uploading successfully.'})
    # Get current dataset.
    else:
        return send_from_directory('./data', 'data-new.csv'), 200


@app.errorhandler(BadRequest)
def handle_bad_request(e):
    return e.description, e.code


if __name__ == '__main__':
    app.run()
