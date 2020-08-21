import firebase_admin
from surprise import dump
from firebase_admin import credentials, storage

cred = credentials.Certificate('./firebase/serviceAccountKey.json')
app = firebase_admin.initialize_app(cred)
bucket = storage.bucket(name='thesis-9b414.appspot.com', app=app)

for file in bucket.list_blobs():
    with open(file, 'wb') as model:
        print(model)