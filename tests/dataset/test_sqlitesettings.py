# This is a minimal test that simply ensures that the object exists,
# basically just to notify us if we do API changes

import qcodes.dataset


def test_settings_exist() -> None:
    limits = qcodes.dataset.SQLiteSettings.limits
    settings = qcodes.dataset.SQLiteSettings.settings

    assert isinstance(limits, dict)
    assert isinstance(settings, dict)
    assert len(limits) == 10
