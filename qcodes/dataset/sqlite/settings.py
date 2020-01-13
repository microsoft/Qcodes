"""
This module takes care of the SQLite settings.
"""
import sqlite3
from typing import Tuple, Dict, Union, Optional


def _read_settings() -> Tuple[Dict[str, Union[str,int]],
                              Dict[str, Union[bool, int, str]]]:
    """
    Function to read the local SQLite settings at import time.

    We mainly care about the SQLite limits, since these play a role
    when committing large amounts of data to the DB, but we record
    everything for good measures.

    Returns:
        Two dictionaries, one with the limits, one with all other settings.
        If a setting has a value, that value is provided. Else a boolean
        indicating wether SQLite was compiled with that option. A missing
        option, say, 'FOOBAR' is equivalent to {'FOOBAR': False}.
    """
    # For the limits, there are known default values
    # (known from https://www.sqlite.org/limits.html)
    DEFAULT_LIMITS: Dict[str, Optional[Union[str, int]]]
    DEFAULT_LIMITS = {'MAX_ATTACHED': 10,
                      'MAX_COLUMN': 2000,
                      'MAX_COMPOUND_SELECT': 500,
                      'MAX_EXPR_DEPTH': 1000,
                      'MAX_FUNCTION_ARG': 100,
                      'MAX_LENGTH': 1_000_000_000,
                      'MAX_LIKE_PATTERN_LENGTH': 50_000,
                      'MAX_PAGE_COUNT': 1_073_741_823,
                      'MAX_SQL_LENGTH': 1_000_000,
                      'MAX_VARIABLE_NUMBER': 999}

    conn = sqlite3.connect(':memory:')
    c = conn.cursor()
    opt_num = 0
    resp = ''

    limits: Dict[str, Optional[Union[str, int]]]
    limits = DEFAULT_LIMITS.copy()
    settings = {}

    c.execute(f'select sqlite_compileoption_get({opt_num});')
    resp = c.fetchone()[0]

    # SQLite only responds back what options have been changed from
    # the default, so we can't know a priori how many responses we'll get
    while resp is not None:
        opt_num += 1
        lst = resp.split('=')
        val: Optional[Union[str, int]]
        if len(lst) == 2:
            (param, val) = lst
            if val.isnumeric():
                val = int(val)
        else:
            param = lst[0]
            val = None

        if param in DEFAULT_LIMITS.keys():
            limits.update({param: val})
        elif val:
            settings.update({param: val})
        else:
            settings.update({param: True})

        c.execute(f'select sqlite_compileoption_get({opt_num});')
        resp = c.fetchone()[0]

    c.execute('select sqlite_version();')
    resp = c.fetchone()[0]
    settings.update({'VERSION': resp})

    c.close()

    return (limits, settings)


class SQLiteSettings:
    """
    Class that holds the machine's sqlite options.

    Note that the settings are not dynamically updated, so changes
    during runtime must be updated manually. But you probably should not
    be changing these settings dynamically in the first place.
    """

    limits, settings = _read_settings()
