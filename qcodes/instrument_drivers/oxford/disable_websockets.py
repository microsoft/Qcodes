import requests

data = {"enable": False}

session = requests.Session()
response = session.post('http://localhost:8000/enablewebsocket', json=data)