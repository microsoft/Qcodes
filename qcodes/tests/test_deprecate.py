from qcodes.utils.deprecate import deprecate


def test_similar_output():

    def _add_one(x):
        return 1 + x


    @deprecate(reason='this function is for private use only.')
    def add_one(x):
        return _add_one(x)

    assert add_one(1) == _add_one(1)
