import pickle
import io
import os
import pandas as pd
import json
from azure.storage.blob import BlobServiceClient, BlobClient
from helper.data import get_azure_ml_stream_from_blob
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# credential = 'rF/htnviSyXuAnJ/XsxCKYUZngsUhC4Heq+BmPufyXqnj25zUde0Qtnrwza0HssD26JkWuqSL39uU1mVnv/6qQ=='
#
# service = BlobServiceClient(account_url="https://quangphamm1902.blob.core.windows.net/", credential=credential)
#
# blob_url = 'https://quangphamm1902.blob.core.windows.net/svd-model/insvd'
# container_name = 'svd-model'
# blob_name = 'insvd'
# blob_client = BlobClient.from_blob_url(blob_url=blob_url, credential=credential)
#
# stream = blob_client.download_blob()
# model = stream.readall()
# blob_to_read = io.BytesIO(model)
# file_content = pickle.load(blob_to_read)
# model = file_content['model']

blob_url = os.getenv('AZURE_ACCOUNT_URL') + '/svd-model/final-new.csv'

blob_client = BlobClient.from_blob_url(blob_url=blob_url, credential=os.getenv('AZUREML_CREDENTIAL'))
stream = blob_client.download_blob()


string_blob = io.BytesIO(stream.readall())
df = pd.read_csv(string_blob, header=0)
print(df.head())
