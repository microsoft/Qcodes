from typing import Callable, List
import re

import pytest
import hypothesis.strategies as hst
from hypothesis import given, settings

from qcodes.dataset.rmq_queue_consumer import QueueConsumer
from qcodes.dataset.rmq_catcher_service import Catcher
from qcodes.dataset.sqlite_storage_interface import SqliteStorageInterface


class MockRMQMessage:
    """
    A mock of the messages that RMQ sends
    """
    def __init__(self, body):
        # TODO: Figure out in how great detail we need to mock
        # channel, method, and properties
        self.ch = 'mock_channel'
        self.method = 'mock_method'
        self.properties = 'mock_properties'
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
    match = re.escape('Received an invalid local_reader_writer, '
                      f'f{bad_writer}, which is not a '
                      'subclass of DataStorageInterface')
    with pytest.raises(ValueError, match=match):
        Catcher(consumer=MockConsumer,
                local_reader_writer=bad_writer)

    bad_consumer = SqliteStorageInterface
    match = re.escape('Received an invalid consumer, '
                      f'f{bad_consumer}, which is not a '
                      'subclass of QueueConsumer')
    with pytest.raises(ValueError, match=match):
        Catcher(consumer=bad_consumer,
                local_reader_writer=bad_writer)

    catcher = Catcher(consumer=MockConsumer,
                      local_reader_writer=SqliteStorageInterface)

    assert catcher.active_guids == []
    assert catcher.number_of_received_messages == 0


@settings(max_examples=10)
@given(N=hst.integers(min_value=1, max_value=1000))
def test_number_of_messages(N):
    # TODO: update the bodies to match the real bodies better
    messages = [MockRMQMessage(body=str(n)) for n in range(N)]

    catcher = Catcher(consumer=MockConsumer,
                      local_reader_writer=SqliteStorageInterface,
                      consumer_kwargs={'messages': messages})

    assert catcher.number_of_received_messages == 0
    catcher.consumer.start_consuming()
    assert catcher.number_of_received_messages == N
