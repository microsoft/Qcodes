
Experimenting with RabbitMQ + Docker

What has kind of worked:

in python venv, pip install pika

in shell:
`docker run --hostname localhost -p 5672:5672 --name some-rabbit rabbitmq:3`

(as far as I can tell, this automatically pulls the image,
i.e. 'rabbitmq' is a repo name and '3' is a tag)

if the name, some-rabbit, is already in use, you can do
`docker start some-rabbit`
to restart the container with whatever settings it was originally started (via `docker run`) with

To see what is currently running, use `docker container ls`

then the following python code works
```python
params = pika.ConnectionParameters('localhost')
conn = pika.BlockingConnection(params)
```

To terminate and remove:
`docker kill some-rabbit`
`docker rm some-rabbit`