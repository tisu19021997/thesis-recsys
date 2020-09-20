import os
from io import StringIO
from azure.storage.blob import BlobClient
from surprise import Dataset, Reader
from sklearn.model_selection import train_test_split


def build_train_test(data_frame, reader, full=True, random_state=42):
    """Returns `surprise` train set and test set from a data frame.
    Args:
        data_frame: Pandas data-frame.
        reader: `surprise` `Reader` instance.
        full: Whether to build full trainset or split one
        random_state (int): RNG seed.
    """
    if full:
        raw_data = Dataset.load_from_df(data_frame, reader=reader)
        train_set = raw_data.build_full_trainset()
        test_set = train_set.build_testset()
        return train_set, test_set

    print(f'Training on 80:20 split')

    #
    if 'timestamp' not in data_frame.columns:
        data_frame['timestamp'] = 0

    train, test = train_test_split(data_frame, test_size=0.2, random_state=random_state)
    print(f'Train set size: {len(train)} / Test set size: {len(test)}')
    data_reader = Dataset(Reader())
    train_set = data_reader.construct_trainset(train.values)
    test_set = data_reader.construct_testset(test.values)
    return train_set, test_set


def get_products_from_ratings(ratings, tolist=False):
    return ratings.asin if not tolist else ratings.asin.values.tolist()


def get_reviewers_from_ratings(ratings, tolist=False):
    return ratings.reviewerID if not tolist else ratings.reviewerID.values.tolist()


def get_azure_ml_stream_from_blob(blob_url, credential=os.getenv('AZUREML_CREDENTIAL')):
    """ Returns the Azure ML stream from a blob URL.
    Args:
        credential (str): Azure Key.
        blob_url (str): Blob URL to fetch.
    """
    blob_client = BlobClient.from_blob_url(blob_url=blob_url, credential=credential)
    stream = blob_client.download_blob()

    return stream.readall()


def is_header_valid(data_header):
    # Check header.
    if data_header != ['u_id', 'i_id', 'rating']:
        return False

    return True
