class AClass:

    def method_a(self) -> None:
        raise RuntimeError('function should not get called')


def a_func() -> None:
    pass
