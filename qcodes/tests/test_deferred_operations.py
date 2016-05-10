from unittest import TestCase

from qcodes.utils.deferred_operations import DeferredOperations


class TestDeferredOperations(TestCase):
    def test_basic(self):
        d = DeferredOperations(lambda: 5)
        self.assertEqual(d(), 5)
        self.assertEqual(d.get(), 5)

    def test_errors(self):
        with self.assertRaises(TypeError):
            DeferredOperations(lambda: 5, args=(1,))

        with self.assertRaises(TypeError):
            DeferredOperations(10)

        # this one doesn't cause errors on definition, only on calling
        d = DeferredOperations(lambda: 1 / 0)
        with self.assertRaises(ZeroDivisionError):
            d()

        # you shouldn't evaluate the truthiness of the DeferredOperations
        # object itself, only after it's called
        d = DeferredOperations(lambda: 5)
        with self.assertRaises(TypeError):
            if d:
                pass

    def test_unary(self):
        d = DeferredOperations(lambda: -3)
        f = DeferredOperations(lambda: 4.221)

        self.assertEqual((abs(d))(), 3)
        self.assertEqual((-d)(), 3)
        self.assertEqual((+d)(), -3)
        self.assertEqual((round(f))(), 4)

    def test_binary_constants(self):
        d = DeferredOperations(lambda: -3)
        f = DeferredOperations(lambda: 4.221)

        self.assertEqual((d == -3)(), True)
        self.assertEqual((d == -4)(), False)
        self.assertEqual((d != -3)(), False)
        self.assertEqual((d != -4)(), True)
        self.assertEqual((d > -4)(), True)
        self.assertEqual((d > -3)(), False)
        self.assertEqual((d >= -3)(), True)
        self.assertEqual((d >= -2)(), False)
        self.assertEqual((d < -3)(), False)
        self.assertEqual((d < -2)(), True)
        self.assertEqual((d <= -3)(), True)
        self.assertEqual((d <= -4)(), False)
        self.assertEqual((d + 5)(), 2)
        self.assertEqual((d & 10)(), 10)
        self.assertEqual((d | 10)(), -3)
        self.assertEqual((d // 2)(), -2)
        self.assertEqual((d % 5)(), 2)
        self.assertEqual((d * 4)(), -12)
        self.assertEqual((d ** 3)(), -27)
        self.assertEqual((d - 10)(), -13)
        self.assertEqual((d / 2)(), -1.5)
        self.assertEqual((7 + d)(), 4)
        self.assertEqual((7 - d)(), 10)
        self.assertEqual((7 * d)(), -21)
        self.assertEqual((1.5 / d)(), -0.5)
        self.assertEqual((7 // d)(), -3)
        self.assertEqual((7 % d)(), -2)
        self.assertEqual((10 ** d)(), 0.001)
        self.assertEqual((8 & d)(), -3)
        self.assertEqual((8 | d)(), 8)
        self.assertEqual((round(f, 1))(), 4.2)

    def test_binary_both(self):
        d4 = DeferredOperations(lambda: 4)
        d5 = DeferredOperations(lambda: 5)

        self.assertEqual((d4 == d5)(), False)
        self.assertEqual((d4 != d5)(), True)
        self.assertEqual((d4 > d5)(), False)
        self.assertEqual((d4 >= d5)(), False)
        self.assertEqual((d4 < d5)(), True)
        self.assertEqual((d4 <= d5)(), True)
        self.assertEqual((d4 + d5)(), 9)
        self.assertEqual((d4 & d5)(), 5)
        self.assertEqual((d4 | d5)(), 4)
        self.assertEqual((d4 // d5)(), 0)
        self.assertEqual((d4 % d5)(), 4)
        self.assertEqual((d4 * d5)(), 20)
        self.assertEqual((d4 ** d5)(), 1024)
        self.assertEqual((d4 - d5)(), -1)
        self.assertEqual((d4 / d5)(), 0.8)
        self.assertEqual((round(d4, d5))(), 4)

    def test_complicated(self):
        d2 = DeferredOperations(lambda: 2)
        d3 = DeferredOperations(lambda: 3)

        self.assertEqual((d2 + (d2 + d2))(), 6)
        self.assertEqual(((d2 < 5) & (d3 > 1))(), True)
        self.assertEqual(((d2 + 5) ** (d3 - d2 + 1))(), 49)
        self.assertEqual((2 * d2 * d3 * d3 * d2 * 5)(), 360)
        self.assertEqual(((1 / d3) < (1 / d2))(), True)
