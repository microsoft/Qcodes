from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import pandas as pd


class DataLengthException(Exception):
    pass


class DataPathException(Exception):
    pass


def dataframe_to_csv(
    dfdict: Mapping[str, pd.DataFrame],
    path: str | Path,
    single_file: bool = False,
    single_file_name: str | None = None,
) -> None:
    import pandas as pd

    dfs_to_save = list()
    for parametername, df in dfdict.items():
        if not single_file:
            dst = os.path.join(path, f"{parametername}.dat")
            df.to_csv(path_or_buf=dst, header=False, sep="\t")
        else:
            dfs_to_save.append(df)
    if single_file:
        df_length = len(dfs_to_save[0])
        if any(len(df) != df_length for df in dfs_to_save):
            raise DataLengthException(
                "You cannot concatenate data "
                "with different length to a "
                "single file."
            )
        if single_file_name is None:
            raise DataPathException(
                "Please provide the desired file name for the concatenated data."
            )
        else:
            if not single_file_name.lower().endswith((".dat", ".csv", ".txt")):
                single_file_name = f"{single_file_name}.dat"
            dst = os.path.join(path, single_file_name)
            df_to_save = pd.concat(dfs_to_save, axis=1)
            df_to_save.to_csv(path_or_buf=dst, header=False, sep="\t")
