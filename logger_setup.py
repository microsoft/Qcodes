# NOTE(giulioungaretti) runme with:
# > python logger_setup.py build_exe (on win)
from cx_Freeze import setup, Executable
import zmq.libzmq
import sys

build_exe_options = {
    # zmq.backend.cython seems to be left out by default
    'packages': ['zmq.backend.cython', ],
    # libzmq.pyd is a vital dependency
    'include_files': [zmq.libzmq.__file__, ],
}
base = None
if sys.platform == "win32":
    base = "Win32GUI"
setup(
    name='zmqlooger',
    version='0.0.1',
    description='zmq logger broker xpub/xsub',
    options={'build_exe': build_exe_options},
    executables=[Executable('qcodes/utils/logger_server.py', base=base)],
)