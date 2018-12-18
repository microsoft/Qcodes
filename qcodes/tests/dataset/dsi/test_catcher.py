from typing import Callable, List, Dict
import re

import pytest
import hypothesis.strategies as hst
from hypothesis import given, settings

from qcodes.dataset.rmq_queue_consumer import QueueConsumer
from qcodes.dataset.rmq_catcher_service import Catcher
from qcodes.dataset.sqlite_storage_interface import (SqliteReaderInterface,
                                                     SqliteWriterInterface)
from qcodes.tests.dataset.temporary_databases import empty_temp_db

class MockChannel:
    """
    A mock of the channel part of a message
    """
    def __init__(self):
        self.acks = 0
        self.nacks = 0

    def basic_ack(self, delivery_tag=0):
        self.acks += 1

    def basic_nack(self, delivery_tag=0, requeue=True):
        self.nacks += 1


class MockProperties:
    """
    A mock of the properties part of a message
    """
    def __init__(self, header: Dict):
        self.headers: Dict = header


class MockMethod:
    """
    A mock of the method part of a message
    """
    def __init__(self):
        self.delivery_tag = 0


class MockRMQMessage:
    """
    A mock of the messages that RMQ sends
    """
    def __init__(self, ch, header, body):
        # TODO: Figure out in how great detail we need to mock
        # channel, method, and properties
        self.ch = ch
        self.method = MockMethod()
        self.properties = MockProperties(header=header)
        self.body = body


class MockConsumer(QueueConsumer):
    """
    Class to mock the consumption of a (RabbitMQ) queue. This mock will
    play the messages it is instantiated with, i.e. pull the all off of the
    imaginary queue.
    """

    def __init__(self, callback: Callable, messages=List[MockRMQMessage]):
        super().__init__(callback)
        self.messages = messages

    def close(self):
        pass

    def start_consuming(self):
        for message in self.messages:
            self.callback(message.ch,
                          message.method,
                          message.properties,
                          message.body)


def test_catcher_init():

    bad_writer = MockConsumer
    match = re.escape('Received an invalid local storage writer, '
                      f'{bad_writer}, which is not a '
                      'subclass of DataWriterInterface')
    with pytest.raises(ValueError, match=match):
        Catcher(consumer=MockConsumer,
                writer=bad_writer)

    bad_consumer = SqliteReaderInterface
    match = re.escape('Received an invalid consumer, '
                      f'{bad_consumer}, which is not a '
                      'subclass of QueueConsumer')
    with pytest.raises(ValueError, match=match):
        Catcher(consumer=bad_consumer,
                writer=bad_writer)

    catcher = Catcher(consumer=MockConsumer)

    assert catcher.active_guids == []
    assert catcher.number_of_received_messages == 0


@settings(max_examples=10, deadline=500)
@given(N=hst.integers(min_value=1, max_value=100))
def test_number_of_messages(empty_temp_db, N):
    # Send a lot of data messages through so that nothing is actually created
    # on disk. Assert that theu all went through and were nack'ed

    channel = MockChannel()

    header = {'guid': 'mock', 'messagetype': 'data'}
    messages = [MockRMQMessage(ch=channel, body=str(n), header=header) for n in range(N)]

    catcher = Catcher(consumer=MockConsumer,
                      consumer_kwargs={'messages': messages})

    assert catcher.number_of_received_messages == 0
    catcher.consumer.start_consuming()
    assert catcher.number_of_received_messages == N
    assert channel.nacks == N
