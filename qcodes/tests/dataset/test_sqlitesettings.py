# This is a minimal test that simply ensures that the object exists,
# basically just to notify us if we do API changes

import qcodes as qc


def test_settings_exist():
    limits = qc.SQLiteSettings.limits
    settings = qc.SQLiteSettings.settings

    assert isinstance(limits, dict)
    assert isinstance(settings, dict)
    assert len(limits) == 10
