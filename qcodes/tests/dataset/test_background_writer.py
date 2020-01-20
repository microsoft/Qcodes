import pytest
import numpy as np
from hypothesis import given, strategies as hst

import qcodes as qc
from qcodes.dataset.measurements import DataSaver
from qcodes.dataset.descriptions.param_spec import ParamSpecBase
from qcodes.dataset.descriptions.dependencies import InterDependencies_
# pylint: disable=unused-import
from qcodes.tests.dataset.temporary_databases import empty_temp_db, experiment

@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize('write_in_background', [False, True])
def test_writing(write_in_background):
    p = ParamSpecBase("p", "text")

    test_set = qc.new_data_set("test-dataset")
    test_set.set_interdependencies(InterDependencies_(standalones=(p,)))
    test_set.mark_started()

    idps = InterDependencies_(standalones=(p,))

    data_saver = DataSaver(
        dataset=test_set, write_period=0, interdeps=idps,
        write_in_background=write_in_background)
    try:
        # data_saver.add_result((p.name, 'yolo'))
        with pytest.raises(ValueError):
            data_saver.add_result((p.name, 2.3))
    finally:
        test_set.mark_completed()
        test_set.conn.close()
