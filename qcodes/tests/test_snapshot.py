"""
Test module for snapshots of instruments and parameters
"""

import pytest

from qcodes.tests.instrument_mocks import (DummyInstrument,
                                           SnapShotTestInstrument)


def test_snapshot_skip_params_update(request):
    """
    Test that params_to_skip_update works as expected, in particular that only
    the parameters given by that variable are skipped
    """

    params = ['v1', 'v2', 'v3', 'v4']


    inst = SnapShotTestInstrument('inst',
                                  params=params,
                                  params_to_skip=[])

    request.addfinalizer(inst.close)

    for param_to_skip in params:

        inst._params_to_skip = param_to_skip
        inst.reset_counter()

        assert list(inst._get_calls.values()) == [0, 0, 0, 0]

        inst.snapshot(update=False)

        assert list(inst._get_calls.values()) == [0, 0, 0, 0]

        inst.snapshot(update=True)

        expected_list = [1, 1, 1, 1]
        expected_list[params.index(param_to_skip)] = 0

        assert list(inst._get_calls.values()) == expected_list
