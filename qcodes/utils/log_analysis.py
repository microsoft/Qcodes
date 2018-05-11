# module for reading logfiles and doing some analysis on them

import dateutil
from typing import Optional, List

import pandas as pd


def logfile_to_dataframe(logfile: Optional[str]=None,
                         columns: Optional[List[str]]=None,
                         separator: Optional[str]=None) -> pd.DataFrame:

    # the expected defaults

    separator = separator if separator else ' - '
    if not columns:
        columns = ['time', 'module', 'function', 'loglevel', 'message']

    with open(logfile) as f:
        raw_cont = f.readlines()

    # note: if we used commas as separators, pandas read_csv
    # could be faster than this string comprehension

    split_cont = [line.split(separator) for line in raw_cont
                  if line[0].isdigit()]  # avoid tracebacks
    dataframe = pd.DataFrame(split_cont, columns=columns)

    return dataframe


def time_difference(time1: str, time2: str) -> float:
    """
    Get the time difference in seconds between two timestamps
    """
    t1 = dateutil.parser.parse(time1)
    t2 = dateutil.parser.parse(time2)

    return (t2 - t1).total_seconds()
