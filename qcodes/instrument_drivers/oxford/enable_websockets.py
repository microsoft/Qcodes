import requests

data = {"enable": True}

session = requests.Session()
response = session.post('http://localhost:8000/enablewebsocket', json=data)