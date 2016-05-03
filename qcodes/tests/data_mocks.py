import multiprocessing as mp

from qcodes.data.data_array import DataArray
from qcodes.data.data_set import new_data


class MockDataManager:
    query_lock = mp.RLock()

    def __init__(self):
        self.needs_restart = False

    def ask(self, *args, timeout=None):
        if args == ('get_data', 'location'):
            return self.location
        elif args == ('get_data',):
            return self.live_data
        elif args[0] == 'new_data' and len(args) == 2:
            if self.needs_restart:
                raise AttributeError('data_manager needs a restart')
            else:
                self.data_set = args[1]
        else:
            raise Exception('unexpected query to MockDataManager')

    def restart(self):
        self.needs_restart = False


class MockFormatter:
    def read(self, data_set):
        data_set.has_read_data = True

    def write(self, data_set):
        data_set.has_written_data = True


class FullIO:
    def list(self, location):
        return [location + '.whatever']


class EmptyIO:
    def list(self, location):
        return []


class MissingMIO:
    def list(self, location):
        if 'm' not in location:
            return [location + '.whatever']
        else:
            return []


class MockLive:
    arrays = 'whole lotta data'


class MockArray:
    array_id = 'noise'

    def init_data(self):
        self.ready = True


def DataSet1D(location=None):
    # DataSet with one 1D array with 5 points
    x = DataArray(name='x', label='X value', preset_data=(1., 2., 3., 4., 5.))
    y = DataArray(name='y', label='Y value', preset_data=(3., 4., 5., 6., 7.),
                  set_arrays=(x,))
    return new_data(arrays=(x, y), location=location)


def DataSet2D(location=None):
    # DataSet with one 2D array, 2x3 points
    x = DataArray(name='x', label='X', preset_data=(5., 6.))
    y = DataArray(name='y', label='Y', preset_data=(7., 8., 9.))
    y.nest(2, 0, x)
    z = DataArray(name='z', label='Z',
                  preset_data=((10., 11., 12.), (13., 14., 15.)),
                  set_arrays=(x, y))
    return new_data(arrays=(x, y, z), location=location)


def DataSetCombined(location=None):
    # Complex DataSet with two 1D and two 2D arrays
    x = DataArray(name='x', label='X!', preset_data=(16., 17.))
    y1 = DataArray(name='y1', label='Y1', preset_data=(18., 19.),
                   set_arrays=(x,))
    y2 = DataArray(name='y2', label='Y2', preset_data=(20., 21.),
                   set_arrays=(x,))

    yset = DataArray(name='yset', label='Y', preset_data=(22., 23., 24.))
    yset.nest(2, 0, x)
    z1 = DataArray(name='z1', label='Z1',
                   preset_data=((25., 26., 27.), (28., 29., 30.)),
                   set_arrays=(x, yset))
    z2 = DataArray(name='z2', label='Z2',
                   preset_data=((31., 32., 33.), (34., 35., 36.)),
                   set_arrays=(x, yset))
    return new_data(arrays=(x, y1, y2, yset, z1, z2), location=location)
