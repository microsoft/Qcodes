"""Standard location_provider class(es) for creating DataSet locations."""
from datetime import datetime
import re
import string

import qcodes.config

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
    location string, based on a format string ``fmt`` and a dict ``record`` of
    information to pass to ``fmt``.

    Default record items are ``date``, ``time``, and ``counter``
    Record item priority from lowest to highest (double items will be
    overwritten):

    - current ``date``, and ``time``
    - record dict from ``__init__``
    - record dict from ``__call__``
    - automatic ``counter``

    For example if any record dict contains a `date` keyword, it will no longer
    be auto-generated.

    Uses ``io.list`` to search for existing data at a matching location.

    ``counter`` must NOT be provided in the record. If ``fmt`` contains
    '{counter}', we look for existing files matching everything BEFORE this,
    then find the highest counter (integer) among those files and use the next
    value.

    If the format string does not contain ``{counter}`` but the location we
    would return is occupied, we add ``'_{counter}'`` to the end.

    Usage::

        loc_provider = FormatLocation(
            fmt='{date}/#{counter}_{time}_{name}_{label}')
        loc = loc_provider(DiskIO('.'),
                           record={'name': 'Rainbow', 'label': 'test'})
        loc
        > '2016-04-30/#001_13-28-15_Rainbow_test'

    Args:
        fmt (str, optional): a format string that all the other info will be
            inserted into. Default '{date}/{time}', or '{date}/{time}_{name}'
            if there is a ``name`` in the record.

        fmt_date (str, optional): a ``datetime.strftime`` format string,
            should only use the date part. The result will be inserted in
            '{date}' in ``fmt``. Default '%Y-%m-%d'.

        fmt_time (str, optional): a ``datetime.strftime`` format string,
            should only use the time part. The result will be inserted in
            '{time}' in ``fmt``. Default '%H-%M-%S'.

        fmt_counter (str, optional): a format string for the counter (integer)
            which is automatically generated from existing DataSets that the
            io manager can see. Default '{03}'.

        record (dict, optional): A dict of default values to provide when
            calling the location_provider. Values provided later will
            override these values.

    Note:
        Do not include date/time or number formatting in ``fmt`` itself, such
        as '{date:%Y-%m-%d}' or '{counter:03}'
    """

    default_fmt = qcodes.config['core']['default_fmt']

    def __init__(self, fmt=None, fmt_date=None, fmt_time=None,
                 fmt_counter=None, record=None):
        # TODO(giulioungaretti) this should be
        # FormatLocation.default_fmt
        self.fmt = fmt or self.default_fmt
        self.fmt_date = fmt_date or '%Y-%m-%d'
        self.fmt_time = fmt_time or '%H-%M-%S'
        self.fmt_counter = fmt_counter or '{:03}'
        self.base_record = record
        self.formatter = SafeFormatter()

        self.counter = 0
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

        Args:
            io (io manager): where we intend to put the new DataSet.

            record (dict, optional): information to insert in the format string
                Any key provided here will override the default record
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

        self.counter = existing_count + 1
        format_record['counter'] = self.fmt_counter.format(self.counter)
        location = self.formatter.format(loc_fmt, **format_record)

        return location
