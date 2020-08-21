import pickle


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
