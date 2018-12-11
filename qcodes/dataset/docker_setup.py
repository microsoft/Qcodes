# Run this file to ensure that the appropriate RMQ docker container
# is running AND that the exchange and queues are set up
import json
from typing import List

import docker
from docker.models.containers import Container

IMAGE_NAME = 'qcodes-rmq-centerpiece'

if __name__ == '__main__':

    print('Starting RMQ docker image')

    client = docker.from_env()
    containers: List[Container] = client.containers.list(all=True)

    if len(containers) == 0:
        print('No running containers')
    else:
        print('The following containers are already running')
        for container in containers:
            print(f'Image: {container.image}, name: {container.name}')

    our_rabbits = [cont for cont in containers if cont.name == IMAGE_NAME]

    if our_rabbits == []:
        other_rabbits = [c for c in containers if
                         'rabbitmq:3' in c.image.attrs['RepoTags'] ]
        for orab in other_rabbits:
            print(f'Removing unauthorized rabbit {orab.name}')
            if orab.status == 'running':
                orab.kill()
            orab.remove()

        print('Starting our container')
        client.containers.run('rabbitmq:3-management', hostname='localhost',
                              ports={'5672/tcp': 5672,
                                     '15672/tcp': 15672}, name=IMAGE_NAME,
                              detach=True)
    else:
        if our_rabbits[0].status == 'running':
            print('Rabbit already running and ready for use')
        else:
            our_rabbits[0].start()




