from qcodes.instrument.parameter import ManualParameter
from qcodes.sweep import sweep, setter, getter


def test_getter_setter():
    """
    Basic test for the getter and setter
    """
    x = ManualParameter("x")
    y = ManualParameter("y")

    def mget(xv):
        return xv ** 2

    def nget(xv, yv):
        yv ** 2 + xv

    m = ManualParameter("m")
    m.get = lambda: mget(x())

    n = ManualParameter("n")
    n.get = lambda: nget(x(), y())

    @getter([("m", "V"), ("n", "I")])
    def gttr():
        return m(), n()

    @setter([("x", "V"), ("y", "I")])
    def sttr(xval, yval):
        x.set(xval)
        y.set(yval)

    xvals = [0, 1, 2]
    yvals = [1, 2, 3]

    so = sweep(sttr, zip(xvals, yvals))(gttr)

    expected_result = [{"x": xv, "y": yv, "m": mget(xv), "n": nget(xv, yv)}
                       for xv, yv in zip(xvals, yvals)]

    assert list(so) == expected_result


def test_class_method_getter_setter():
    """
    The getter and setter should work when the functions decorated are class
    methods
    """
    class Test:
        def __init__(self):

            self.x = None
            self.y = None

        def mget(self, xv):
            return xv ** 2

        def nget(self, xv, yv):
            yv ** 2 + xv

        @getter([("m", "V"), ("n", "I")])
        def gttr(self):
            return self.mget(self.x), self.nget(self.x, self.y)

        @setter([("x", "V"), ("y", "I")])
        def sttr(self, xval, yval):
            self.x = xval
            self.y = yval

        def test(self):

            xvals = [0, 1, 2]
            yvals = [1, 2, 3]

            so = sweep(self.sttr, zip(xvals, yvals))(self.gttr)

            expected_result = [
                {"x": xv, "y": yv, "m": self.mget(xv), "n": self.nget(xv, yv)}
                for xv, yv in zip(xvals, yvals)]

            assert list(so) == expected_result

    Test().test()


def test_layout_info_setter():

    @setter([("BB", "V"), ("AB", "V")])
    def setter_function1(v1, v2):
        return

    @setter(
        [("BB", "V"), ("AB", "V")],
        inferred_parameters=[("bb", "mV"), ("ab", "mV")]
    )
    def setter_function2(v1, v2):
        return 0, 0

    m = ManualParameter("m")

    s1 = [0, 1, 2]
    s2 = [1, 3, 5]
    sweep_values = list(zip(s1, s2))

    so = sweep(setter_function1, sweep_values)(m)
    assert so.parameter_table.layout_info("m") == {
        "BB": {
            "min": min(s1),
            "max": max(s1),
            "length": len(s1),
            "steps": s1[1] - s1[0]
        },
        "AB": {
            "min": min(s2),
            "max": max(s2),
            "length": len(s2),
            "steps": s2[1] - s2[0]
        }
    }

    so = sweep(setter_function2, sweep_values)(m)
    assert so.parameter_table.layout_info("m") == {
        "bb": {
            "min": "?",
            "max": "?",
            "length": len(s1),
            "steps": "?"
        },
        "ab": {
            "min": "?",
            "max": "?",
            "length": len(s2),
            "steps": "?"
        }
    }
