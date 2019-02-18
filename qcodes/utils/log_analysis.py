# module for reading logfiles and doing some analysis on them

from typing import Optional, List

import pandas as pd
from pandas.core.series import Series

from qcodes.utils.deprecate import deprecate


@deprecate(reason="The logging infrastructure has moved to `qcodes.utils.logger`",
           alternative="`qcodes.utils.logger.logfile_to_dataframe`")
def logfile_to_dataframe(logfile: str,
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


@deprecate(reason="The logging infrastructure has moved to `qcodes.utils.logger`",
           alternative="`qcodes.utils.logger.time_difference`")
def time_difference(firsttimes: Series,
                    secondtimes: Series,
                    use_first_series_labels: bool=True) -> Series:
    """
    Calculate the time differences between two series
    containing time stamp strings as their values.

    Args:
        firsttimes: The oldest times
        secondtimes: The youngest times
        use_first_series_labels: If true, the returned Series
            has the same labels as firsttimes. Else it has
            the labels of secondtimes

    Returns:
        A Series with float values of the time difference (s)
    """

    if ',' in firsttimes.iloc[0]:
        nfirsttimes = firsttimes.str.replace(',', '.')
    else:
        nfirsttimes = firsttimes

    if ',' in secondtimes.iloc[0]:
        nsecondtimes = secondtimes.str.replace(',', '.')
    else:
        nsecondtimes = secondtimes

    t0s = nfirsttimes.astype("datetime64[ns]")
    t1s = nsecondtimes.astype("datetime64[ns]")
    timedeltas = (t1s.values - t0s.values).astype('float')*1e-9

    if use_first_series_labels:
        output = pd.Series(timedeltas, index=nfirsttimes.index)
    else:
        output = pd.Series(timedeltas, index=nsecondtimes.index)

    return output
