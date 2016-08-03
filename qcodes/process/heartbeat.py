import os
import mmap

from qcodes import config


def makeHeartBeatFile(bfile, reinit=False):
    if (not os.path.exists(bfile)) or reinit:
        f = open(bfile, 'wb')
        f.write(bytes([99, 1]))  # heartbeat on
        f.close()


def initHeartBeat(bfile, reinit=False):
    ''' Initialize connection to heartbeat for writing, use with setHeartBeat '''
    if (not os.path.exists(bfile)) or reinit:
        f = open(bfile, 'wb')
        f.write(bytes([99, 1]))  # heartbeat on
        f.close()

    f = open(bfile, 'a+b')
    f.seek(0)
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
    m.flush()
    # print(m[0])
    # print(m[1])
    # print('size %d' % m.size() )

    return m


def openHeartBeat(bfile):
    f = open(bfile, 'r')
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return m


def setHeartBeat(m, value):
    ''' Set heartbeat value '''
    m[1] = value


def readHeartBeat(m):
    ''' Return heartbeat value '''
    return m[1]


_bfile = config.heartbeatfile
makeHeartBeatFile(_bfile)
