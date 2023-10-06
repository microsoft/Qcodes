"""This module defines the ExportInfo dataclass."""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass

from qcodes.dataset.export_config import DataExportType


@dataclass
class ExportInfo:

    export_paths: dict[str, str]

    def __post_init__(self) -> None:
        """Verify that keys used in export_paths are as expected."""
        allowed_keys = tuple(a.value for a in DataExportType)
        for key in self.export_paths.keys():
            if key not in allowed_keys:
                warnings.warn(
                    f"The supported export types are: {allowed_keys}. Got {key} "
                    f"which is not supported"
                )

    def to_str(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_str(cls, string: str) -> ExportInfo:
        if string == "":
            return cls({})

        datadict = json.loads(string)
        return cls(**datadict)
