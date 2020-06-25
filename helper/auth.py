import requests
from auth.auth import BearerAuth


def is_good_request(request):
    _, jwt = request.headers['Authorization'].split()
    return is_good_token(jwt)


def is_good_token(json_web_token):
    r = requests.get('http://127.0.0.1:8081/api/v1/auth/admin', auth=BearerAuth(json_web_token))
    return r.status_code == 200
