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

    def read_metadata(self, data_set):
        data_set.has_read_metadata = True

    def write_metadata(self, data_set):
        data_set.has_written_metadata = True


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
    x = DataArray(name='x', label='X', preset_data=(1., 2., 3., 4., 5.))
    y = DataArray(name='y', label='Y', preset_data=(3., 4., 5., 6., 7.),
                  set_arrays=(x,))
    return new_data(arrays=(x, y), location=location)


def file_1d():
    return '\n'.join([
        '# x\ty',
        '# "X"\t"Y"',
        '# 5',
        '1\t3',
        '2\t4',
        '3\t5',
        '4\t6',
        '5\t7', ''])


def DataSetCombined(location=None):
    # Complex DataSet with two 1D and two 2D arrays
    x = DataArray(name='x', label='X!', preset_data=(16., 17.))
    y1 = DataArray(name='y1', label='Y1 value', preset_data=(18., 19.),
                   set_arrays=(x,))
    y2 = DataArray(name='y2', label='Y2 value', preset_data=(20., 21.),
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


def files_combined():
    return [
        '\n'.join([
            '# x\ty1\ty2',
            '# "X!"\t"Y1 value"\t"Y2 value"',
            '# 2',
            '16\t18\t20',
            '17\t19\t21', '']),

        '\n'.join([
            '# x\tyset\tz1\tz2',
            '# "X!"\t"Y"\t"Z1"\t"Z2"',
            '# 2\t3',
            '16\t22\t25\t31',
            '16\t23\t26\t32',
            '16\t24\t27\t33',
            '',
            '17\t22\t28\t34',
            '17\t23\t29\t35',
            '17\t24\t30\t36', ''])
    ]
