
import os
import tempfile

addzmqlogging = os.environ.get('ZMQLOGGING', 1)

heartbeatfile  = os.path.join(tempfile.gettempdir(), r'qcodes-heartbeat.bin' )
