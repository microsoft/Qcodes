from datetime import datetime
from unittest import TestCase

from qcodes.data.location import FormatLocation, SafeFormatter

from .data_mocks import MatchIO


class TestSafeFormatter(TestCase):
    def test_normal_formatting(self):
        formatter = SafeFormatter()
        self.assertEqual(formatter.format('{}{{{}}}', 1, 2), '1{2}')
        self.assertEqual(formatter.format('{apples}{{{oranges}}}',
                                          apples='red', oranges='unicorn'),
                         'red{unicorn}')

    def test_missing(self):
        formatter = SafeFormatter()
        self.assertEqual(formatter.format('{}'), '{0}')
        self.assertEqual(formatter.format('{cheese}', fruit='pears'),
                         '{cheese}')


def _default(time: datetime, formatter: FormatLocation, counter:str, name: str):
    date = time.strftime(formatter.fmt_date)
    mytime = time.strftime(formatter.fmt_time)
    fmted = formatter.formatter.format(formatter.default_fmt,
                                       date=date,
                                       counter=counter,
                                       time=mytime,
                                       name=name)
    return fmted


class TestFormatLocation(TestCase):
    def test_default(self):
        lp = FormatLocation()
        fmt = '%Y-%m-%d/%H-%M-%S'

        name = "name"
        self.assertEqual(lp(MatchIO([]), {'name': name}),
                         _default(datetime.now(), lp, "001", name))

        # counter starts at +1  using MatchIo undocumented magic argument
        start_magic_value = 5
        self.assertEqual(
            lp(MatchIO(["", f"{start_magic_value:03d}"]), {"name": name}),
            _default(datetime.now(), lp, f"{start_magic_value+1:03d}", name),
        )

    def test_fmt_subparts(self):
        lp = FormatLocation(fmt='{date}/{time}', fmt_date='%d-%b-%Y', fmt_time='%I-%M%p',
                            fmt_counter='##{:.1f}~')
        fmt = '%d-%b-%Y/%I-%M%p'

        self.assertEqual(lp(MatchIO([])),
                         datetime.now().strftime(fmt))
        self.assertEqual(lp(MatchIO([]), {'name': 'who?'}),
                         datetime.now().strftime(fmt) + '_who?')

        self.assertEqual(lp(MatchIO([''])),
                         datetime.now().strftime(fmt) + '_##2.0~')
        self.assertEqual(lp(MatchIO(['', '9'])),
                         datetime.now().strftime(fmt) + '_##10.0~')
        self.assertEqual(lp(MatchIO(['', '12345']), {'name': 'you!'}),
                         datetime.now().strftime(fmt) + '_you!_##12346.0~')

    def test_record_call(self):
        lp = FormatLocation(fmt='{date}/#{counter}_{time}_{name}_{label}')

        # counter starts at 1 if it's a regular part of the format string
        expected = datetime.now().strftime('%Y-%m-%d/#001_%H-%M-%S_Joe_fruit')
        self.assertEqual(lp(MatchIO([]), {'name': 'Joe', 'label': 'fruit'}),
                         expected)

        expected = datetime.now().strftime('%Y-%m-%d/#1000_%H-%M-%S_Ga_As')
        self.assertEqual(lp(MatchIO(['999']), {'name': 'Ga', 'label': 'As'}),
                         expected)

        # missing label
        expected = datetime.now().strftime(
            '%Y-%m-%d/#001_%H-%M-%S_Fred_{label}')
        self.assertEqual(lp(MatchIO([]), {'name': 'Fred'}), expected)

    def test_record_override(self):
        # this one needs 'c' filled in at call time
        lp = FormatLocation(fmt='{a}_{b}_{c}', record={'a': 'A', 'b': 'B'})
        io = MatchIO([])

        self.assertEqual(lp(io, {'c': 'C'}), 'A_B_C')
        self.assertEqual(lp(io, {'a': 'aa', 'c': 'cc'}), 'aa_B_cc')
        # extra keys just get discarded
        self.assertEqual(lp(io, {'c': 'K', 'd': 'D'}), 'A_B_K')

        self.assertEqual(lp(MatchIO([])), 'A_B_{c}')
        self.assertEqual(lp(MatchIO([]), {'a': 'AA'}), 'AA_B_{c}')

        # this one has defaults for everything, so nothing is needed at
        # call time, but things can still be overridden
        lp = FormatLocation(fmt='{d}_{e}', record={'d': 'D', 'e': 'E'})

        self.assertEqual(lp(MatchIO([])), 'D_E')
        self.assertEqual(lp(MatchIO([]), {'d': 'T'}), 'T_E')

    def test_errors(self):
        io = MatchIO([])

        # gives the wrong integer (extra 0 at the end!)
        with self.assertRaises(ValueError):
            FormatLocation(fmt_counter='{:03}0')

        # you're not allowed to give a counter. Whatcha trying to do anyway?
        # I'm tempted to say the same about time and date, but will leave them
        # overridable for now.
        with self.assertRaises(KeyError):
            FormatLocation()(io, {'counter': 100})
        with self.assertRaises(KeyError):
            FormatLocation(record={'counter': 100})(io)
