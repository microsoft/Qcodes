from datetime import datetime
import re
import string


class SafeFormatter(string.Formatter):

    """Modified string formatter that doesn't complain about missing keys."""

    def get_value(self, key, args, kwargs):
        """Missing keys just get left as they were: '{key}'."""
        try:
            return super().get_value(key, args, kwargs)
        except:
            return '{{{}}}'.format(key)


class FormatLocation:

    """
    This is the default DataSet Location provider.

    It provides a callable that returns a new (not used by another DataSet)
    location string, based on a format string `fmt` and a dict `record` of
    information to pass to `fmt`.

    A base record can be provided on instantiating the FormatLocation, and
    another one can be provided on call, which overrides parts of the base
    just for that call.

    Default record items are `{date}`, `{time}`, and `{counter}`
    Record item priority from lowest to highest (double items will be
    overwritten)
    - `{counter}`, `{date}`, `{time}`
    - records dict from `__init__`
    - records dict from `__call__`
    Thus if any record dict contains a `date` keyword, it will no longer be
    auto-generated.

    Uses `io.list` to search for existing data at a matching location.

    `{counter}` is special and must NOT be provided in the record.
    If the format string contains `{counter}`, we look for existing files
    matching everything before the counter, then find the highest counter
    (integer) among those files and use the next value. That means the counter
    only increments as long as fields before it do not change, and files with
    incrementing counters will always group together and sort correctly in
    directory listings

    If the format string does not contain `{counter}` but the location we would
    return is occupied, we will add '_{counter}' to the end and do the same.

    Usage:
    ```
        loc_provider = FormatLocation(
            fmt='{date}/#{counter}_{time}_{name}_{label}')
        loc = loc_provider(DiskIO('.'),
                           record={'name': 'Rainbow', 'label': 'test'})
        loc
        > '2016-04-30/#001_13-28-15_Rainbow_test'
    ```
    Default format string is '{date}/{time}', and if `name` exists in record,
    it is '{date}/{time}_{name}'
    with `fmt_date='%Y-%m-%d'` and `fmt_time='%H-%M-%S'`
    """

    default_fmt = '{date}/{time}'

    def __init__(self, fmt=None, fmt_date=None, fmt_time=None,
                 fmt_counter=None, record=None):
        """
        Construct a FormatLocation location_provider.

        fmt (default '{date}/{time}'): a format string that all the other info
            will get inserted into

        fmt_date (default '%Y-%m-%d'): a `datetime.strftime` format string,
            accepts datetime.now() but should only use the date part. The
            result will be inserted in '{date}' in `fmt`. Do not include
            date formatting in `fmt` itself (such as '{date:%Y-%m-%d}')

        fmt_time (default '%H-%M-%S'): a `datetime.strftime` format string,
            accepts datetime.now() but should only use the time part. The
            result will be inserted in '{time}' in `fmt`. Do not include
            date formatting in `fmt` itself (such as '{time:%H-%M-%S}')

        fmt_counter (default '{:03}'): a format string for the counter
            (integer) which is automatically generated from existing
            DataSets that the io manager can see. Do not include
            number formatting in `fmt` itself (such as '{counter:03}')

        record (default None): a dict of default values to provide when
            calling the location_provider. Values provided later will
            override these values.
        """
        self.fmt = fmt or self.default_fmt
        self.fmt_date = fmt_date or '%Y-%m-%d'
        self.fmt_time = fmt_time or '%H-%M-%S'
        self.fmt_counter = fmt_counter or '{:03}'
        self.base_record = record
        self.formatter = SafeFormatter()

        for testval in (1, 23, 456, 7890):
            if self._findint(self.fmt_counter.format(testval)) != testval:
                raise ValueError('fmt_counter must produce a correct integer '
                                 'representation of its argument (eg "{:03}")',
                                 fmt_counter)

    def _findint(self, s):
        try:
            return int(re.findall(r'\d+', s)[0])
        except:
            return 0

    def __call__(self, io, record=None):
        """
        Call the location provider to get a new location.

        io: an io manager instance

        record (default None): a dict of information to use in the
            format string
        """
        loc_fmt = self.fmt

        time_now = datetime.now()
        date = time_now.strftime(self.fmt_date)
        time = time_now.strftime(self.fmt_time)
        format_record = {'date': date, 'time': time}

        if self.base_record:
            format_record.update(self.base_record)
        if record:
            format_record.update(record)

        if 'counter' in format_record:
            raise KeyError('you must not provide a counter in your record.',
                           format_record)

        if ('name' in format_record) and ('{name}' not in loc_fmt):
            loc_fmt += '_{name}'

        if '{counter}' not in loc_fmt:
            location = self.formatter.format(loc_fmt, **format_record)
            if io.list(location):
                loc_fmt += '_{counter}'
                # redirect to the counter block below, but starting from 2
                # because the already existing file counts like 1
                existing_count = 1
            else:
                return location
        else:
            # if counter is already in loc_fmt, start from 1
            existing_count = 0

        # now search existing files for the next allowed counter

        head_fmt = loc_fmt.split('{counter}', 1)[0]
        # io.join will normalize slashes in head to match the locations
        # returned by io.list
        head = io.join(self.formatter.format(head_fmt, **format_record))

        file_list = io.list(head + '*', maxdepth=0, include_dirs=True)

        for f in file_list:
            cnt = self._findint(f[len(head):])
            existing_count = max(existing_count, cnt)

        format_record['counter'] = self.fmt_counter.format(existing_count + 1)
        location = self.formatter.format(loc_fmt, **format_record)

        return location
