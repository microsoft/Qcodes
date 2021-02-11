import numpy

from qcodes.data.data_array import DataArray
from qcodes.data.data_set import new_data
from qcodes.data.io import DiskIO


class MockFormatter:
    def read(self, data_set):
        data_set.has_read_data = True

    def write(self, data_set, io_manager, location, write_metadata=False):
        data_set.has_written_data = True

    def read_metadata(self, data_set):
        data_set.has_read_metadata = True

    def write_metadata(self, data_set, io_manager, location, read_first=True):
        data_set.has_written_metadata = True


class RecordingMockFormatter:
    def __init__(self):
        self.write_calls = []
        self.modified_ranges = []
        self.last_saved_indices = []
        self.write_metadata_calls = []

    def write(self, data_set, io_manager, location, force_write=False):
        self.write_calls.append((io_manager.base_location, location))

        self.modified_ranges.append({
            array_id: array.modified_range
            for array_id, array in data_set.arrays.items()
        })

        self.last_saved_indices.append({
            array_id: array.last_saved_index
            for array_id, array in data_set.arrays.items()
        })

    def write_metadata(self, data_set, io_manager, location, read_first=True):
        self.write_metadata_calls.append((io_manager.base_location,
                                          location, read_first))


class MatchIO:
    def __init__(self, existing_matches, fmt=None):
        self.existing_matches = existing_matches
        self.fmt = fmt or '{}{}.something'

    def list(self, location, **kwargs):
        return [self.fmt.format(location, i) for i in self.existing_matches]

    def join(self, *args):
        return DiskIO('.').join(*args)


class MockLive:
    arrays = 'whole lotta data'


class MockArray:
    array_id = 'noise'

    def init_data(self):
        self.ready = True


def DataSet1D(location=None, name=None):
    # DataSet with one 1D array with 5 points

    # TODO: since y lists x as a set_array, it should automatically
    # set is_setpoint=True for x, shouldn't it? Any reason we woundn't
    # want that?
    x = DataArray(name='x', label='X', preset_data=(1., 2., 3., 4., 5.),
                  is_setpoint=True)
    y = DataArray(name='y', label='Y', preset_data=(3., 4., 5., 6., 7.),
                  set_arrays=(x,))
    return new_data(arrays=(x, y), location=location, name=name)


def DataSet2D(location=None, name=None):
    # DataSet with one 2D array with 4 x 6 points
    yy, xx = numpy.meshgrid(range(4), range(6))
    zz = xx**2+yy**2
    # outer setpoint should be 1D
    xx = xx[:, 0]
    x = DataArray(name='x', label='X', preset_data=xx, is_setpoint=True)
    y = DataArray(name='y', label='Y', preset_data=yy, set_arrays=(x,),
                  is_setpoint=True)
    z = DataArray(name='z', label='Z', preset_data=zz, set_arrays=(x, y))
    return new_data(arrays=(x, y, z), location=location, name=name)


def file_1d():
    return '\n'.join([
        '# x_set\ty',
        '# "X"\t"Y"',
        '# 5',
        '1\t3',
        '2\t4',
        '3\t5',
        '4\t6',
        '5\t7', ''])


def DataSetCombined(location=None, name=None):
    # Complex DataSet with two 1D and two 2D arrays
    x = DataArray(name='x', label='X!', preset_data=(16., 17.),
                  is_setpoint=True)
    y1 = DataArray(name='y1', label='Y1 value', preset_data=(18., 19.),
                   set_arrays=(x,))
    y2 = DataArray(name='y2', label='Y2 value', preset_data=(20., 21.),
                   set_arrays=(x,))

    yset = DataArray(name='y', label='Y', preset_data=(22., 23., 24.),
                     is_setpoint=True)
    yset.nest(2, 0, x)
    z1 = DataArray(name='z1', label='Z1',
                   preset_data=((25., 26., 27.), (28., 29., 30.)),
                   set_arrays=(x, yset))
    z2 = DataArray(name='z2', label='Z2',
                   preset_data=((31., 32., 33.), (34., 35., 36.)),
                   set_arrays=(x, yset))
    return new_data(arrays=(x, y1, y2, yset, z1, z2), location=location,
                    name=name)


def files_combined():
    return [
        '\n'.join([
            '# x_set\ty1\ty2',
            '# "X!"\t"Y1 value"\t"Y2 value"',
            '# 2',
            '16\t18\t20',
            '17\t19\t21', '']),

        '\n'.join([
            '# x_set\ty_set\tz1\tz2',
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
