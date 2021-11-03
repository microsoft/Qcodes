from unittest import TestCase
import numpy as np

import qcodes
import qcodes.measure
from qcodes.data.hdf5_format import HDF5Format, HDF5FormatMetadata
from qcodes.data.gnuplot_format import GNUPlotFormat
from qcodes.data.data_set import load_data
from qcodes.tests.legacy.data_mocks import DataSet2D


#%%
class TestFormatters(TestCase):

    def setUp(self):
        self.formatters = [GNUPlotFormat, HDF5Format, HDF5FormatMetadata]
        self.metadata = {'subdict': {'stringlist': ['P1']}, 'string': 'P1',
                         'int': 1, 'list': [1, 2], 'numpyarray': np.array([1])}

    def test_read_write(self):
        for f in self.formatters:
            print('test formatter %s' % f)
            dataset = DataSet2D(name="test_read_write")
            dataset.formatter = f()

            dataset.add_metadata(self.metadata)
            dataset.write(write_metadata=True)

            dataset2 = load_data(dataset.location, formatter=f())
            self.assertEqual(list(dataset.arrays.keys()),
                             list(dataset2.arrays.keys()))
            # strings should be read and written identically
            self.assertEqual(dataset.metadata['string'],
                             dataset2.metadata['string'])


class TestNoSorting(TestCase):
    """
    (WilliamHPNielsen): I'm not too sure where this test belongs... It tests
    that parameters with non-sortable keys can be saved using the gnuplot
    formatter, so I guess it goes here.
    """

    param = qcodes.Parameter(name='mixed_val_mapping_param',
                             get_cmd=lambda: np.random.randint(1, 3),
                             val_mapping={1: 1, '2': 2}
                             )

    def test_can_measure(self):
        qcodes.measure.Measure(self.param).run(name="test_no_sorting")
