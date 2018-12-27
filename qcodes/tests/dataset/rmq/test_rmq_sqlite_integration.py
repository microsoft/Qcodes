# integration testing
#
# For now we assume that the rmq_setup.py  --testing=True
# has already been run

from subprocess import Popen, CREATE_NEW_PROCESS_GROUP
import signal
import os

import pytest

from qcodes.tests.dataset.temporary_databases import empty_temp_db
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.sqlite_storage_interface import SqliteReaderInterface
from qcodes.dataset.rabbitmq_storage_interface import RabbitMQWriterInterface
import qcodes.dataset

@pytest.fixture(scope="function", autouse=True)
def catcher_running():

    script_path = qcodes.dataset.__file__.replace('__init__.py',
                                                  'rmq_catcher_service.py')

    # start the catcher service
    proc = Popen(['python', script_path, '--testing=True'],
                 creationflags=CREATE_NEW_PROCESS_GROUP)
    try:
        yield
    finally:
        print('Time for a kill')
        #proc.send_signal(signal.SIGINT)
        os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
        proc.wait()
        #proc.kill()
        print('Killed the damn thing')
        # why are we still blocking here? WHy cant the test exit?

@pytest.fixture(scope="function")
def catcher_running_alt(request):

    script_path = qcodes.dataset.__file__.replace('__init__.py',
                                                  'rmq_catcher_service.py')

    # start the catcher service
    proc = Popen(['python', script_path, '--testing=True'])
    request.addfinalizer(proc.kill)

def test_run_creation(empty_temp_db, request):

    ds = DataSet(writerinterface=RabbitMQWriterInterface,
                 exp_name='my experiment',
                 sample_name='no sample',
                 writer_kwargs={'use_test_queue': True})
    request.addfinalizer(ds.dsi.reader.conn.close)
    request.addfinalizer(ds.dsi.writer.close)
    request.addfinalizer(ds.dsi.writer.channel.close)
    print('Created dataset')
    assert False
