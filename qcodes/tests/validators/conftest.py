class AClass:

    def method_a(self):
        raise RuntimeError('function should not get called')


def a_func():
    pass
