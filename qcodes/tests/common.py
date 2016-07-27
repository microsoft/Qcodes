def strip_qc(d, keys=('instrument', '__class__')):
    # depending on how you run the tests, __module__ can either
    # have qcodes on the front or not. Just strip it off.
    for key in keys:
        if key in d:
            d[key] = d[key].replace('qcodes.tests.', 'tests.')
    return d
