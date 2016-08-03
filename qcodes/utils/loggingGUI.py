"""
A GUI for multi-processing logging using ZMQ

Code is adapted from https://github.com/zeromq/pyzmq/blob/master/examples/logger/zmqlogger.py

Pieter Eendebak <pieter.eendebak@tno.nl>

"""

#%% Import packages
import logging
import os
import signal
import time
import argparse
import re

from qtpy import QtGui
from qtpy import QtWidgets


import zmq
from zmq.log.handlers import PUBHandler

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', default=1, help="verbosity level")
parser.add_argument(
    '-l', '--level', default=logging.DEBUG, help="logging level")
parser.add_argument('-p', '--port', type=int, default=5800, help="zmq port")
parser.add_argument('-g', '--gui', type=int, default=1, help="show gui")
args = parser.parse_args()


#%% Util functions


def static_var(varname, value):
    """ Helper function to create a static variable """
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


@static_var("time", 0)
def tprint(string, dt=1, output=False):
    """ Print progress of a loop every dt seconds """
    if (time.time() - tprint.time) > dt:
        print(string)
        tprint.time = time.time()
        if output:
            return True
        else:
            return
    else:
        if output:
            return False
        else:
            return

#%% Functions for installing the logger


import zmq.log.handlers


def removeZMQlogger(name=None):
    logger = logging.getLogger(name)

    for h in logger.handlers:
        if isinstance(h, zmq.log.handlers.PUBHandler):
            print('removing handler %s' % h)
            logger.removeHandler(h)


def installZMQlogger(port=5800, name=None, clear=True, level=logging.INFO):
    if clear:
        removeZMQlogger(name)

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.RCVHWM, 10)

    pub.connect('tcp://127.0.0.1:%i' % port)

    if name is None:
        rootlogger = logging.getLogger()
    else:
        rootlogger = logging.getLogger(name)
    if level is not None:
        rootlogger.setLevel(level)
    handler = PUBHandler(pub)
    pid = os.getpid()
    pstr = 'pid %d: ' % pid
    handler.formatters = {
        logging.DEBUG: logging.Formatter(pstr +
                                         "%(levelname)s %(filename)s:%(lineno)d - %(message)s\n"),
        logging.INFO: logging.Formatter(pstr + "%(message)s\n"),
        logging.WARN: logging.Formatter(pstr +
                                        "%(levelname)s %(filename)s:%(lineno)d - %(message)s\n"),
        logging.ERROR: logging.Formatter(pstr +
                                         "%(levelname)s %(filename)s:%(lineno)d - %(message)s - %(exc_info)s\n"),
        logging.CRITICAL: logging.Formatter(pstr +
                                            "%(levelname)s %(filename)s:%(lineno)d - %(message)s\n")}

    rootlogger.addHandler(handler)
    logging.debug('installZMQlogger: handler installed')
                  # first message always is discarded
    return rootlogger

#%%


class zmqLoggingGUI(QtWidgets.QDialog):

    LOG_LEVELS = dict({logging.DEBUG: 'debug', logging.INFO: 'info',
                       logging.WARN: 'warning', logging.ERROR: 'error', logging.CRITICAL: 'critical'})

    def __init__(self, parent=None):
        super(zmqLoggingGUI, self).__init__(parent)

        self.setWindowTitle('ZMQ logger')

        self.imap = dict((v, k) for k, v in self.LOG_LEVELS.items())

        self._console = QtWidgets.QPlainTextEdit(self)
        self._console.setMaximumBlockCount(2000)

        self._button = QtWidgets.QPushButton(self)
        self._button.setText('Clear')
        self._killbutton = QtWidgets.QPushButton(self)
        self._killbutton.setText('Kill heartbeat')

        self._levelBox = QtWidgets.QComboBox(self)
        for k in sorted(self.LOG_LEVELS.keys()):
            print('item %s' % k)
            val = self.LOG_LEVELS[k]
            self._levelBox.insertItem(k, val)

        blayout = QtWidgets.QHBoxLayout()
        blayout.addWidget(self._button)
        blayout.addWidget(self._killbutton)
        blayout.addWidget(self._levelBox)
        self._button.clicked.connect(self.clearMessages)
        self._killbutton.clicked.connect(self.killPID)
        self._levelBox.currentIndexChanged.connect(self.setLevel)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self._console)
        layout.addLayout(blayout)
        self.setLayout(layout)

        self.addMessage('logging started...' + '\n')

        self._levelBox.setCurrentIndex(1)
        self.loglevel = logging.INFO
        self.nkill = 0

    def setLevel(self, boxidx):
        name = self._levelBox.itemText(boxidx)
        lvl = self.imap.get(name, None)
        print('set level to %s: %d' % (name, lvl))
        if lvl is not None:
            self.loglevel = lvl

    def addMessage(self, msg, level=None):
        if level is not None:
            if level < self.loglevel:
                return
        self._console.moveCursor(QtGui.QTextCursor.End)
        self._console.insertPlainText(msg)
        self._console.moveCursor(QtGui.QTextCursor.End)

    def clearMessages(self):
        ''' Clear the messages in the logging window '''
        self._console.clear()
        self.addMessage('cleared messages...\n')

    def killPID(self):
        ''' Clear the messages in the logging window '''
        self.nkill = 10


def qt_logger(port, dlg, level=logging.INFO, verbose=1):
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.bind('tcp://127.0.0.1:%i' % port)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVHWM, 10)

    logging.basicConfig(level=level)
    app = QtWidgets.QApplication.instance()
    app.processEvents()

    print('connected to port %s' % port)
    while True:
        tprint('ZMQ logger: logging...', dt=5)
        try:
            level, message = sub.recv_multipart(zmq.NOBLOCK)
            # level, message = sub.recv_multipart()
            message = message.decode('ascii')
            if message.endswith('\n'):
                # trim trailing newline, which will get appended again
                message = message[:-1]
            level = level.lower().decode('ascii')
            log = getattr(logging, level)
            lvlvalue = dlg.imap.get(level, None)

            if verbose >= 2:
                log(message)
            dlg.addMessage(message + '\n', lvlvalue)

            if dlg.nkill > 0:
                print('check pid')
                m = re.match('pid (\d*): heartbeat', message)
                dlg.nkill = dlg.nkill - 1
                if m is not None:
                    pid = int(m.group(1))
                    print('killing pid %d' % pid)
                    os.kill(pid, signal.SIGKILL)  # or signal.SIGKILL
                    dlg.addMessage(
                        'send kill signal to pid %d\n' % pid, logging.CRITICAL)
            app.processEvents()

            if verbose >= 2:
                print('message: %s (level %s)' % (message, level))
        except zmq.error.Again:
            # no messages in system....
            app.processEvents()
            time.sleep(.06)
            message = ''
            level = None

#%%
if __name__ == '__main__':

    port = args.port
    verbose = args.verbose

    app = None
    if (not QtWidgets.QApplication.instance()):
        app = QtWidgets.QApplication([])
    dlg = zmqLoggingGUI()
    dlg.resize(800, 400)
    dlg.show()

    # start the log watcher
    try:
        # sub_logger(port, level=args.level, verbose=verbose)
        qt_logger(port, level=args.level, verbose=verbose, dlg=dlg)
        pass
    except KeyboardInterrupt:
        pass

    # if (app):
    #    app.exec_()

#%% Send message to logger
if 0:
    port = 5800
    installZMQlogger(port=port, level=None)
    logging.warning('test')
    # log_worker(port=5700, interval=1)
