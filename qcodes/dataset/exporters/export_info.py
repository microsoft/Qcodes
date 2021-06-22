"""
This module defines the ExportInfo dataclass
"""
import json
from dataclasses import asdict, dataclass
from typing import Dict

from qcodes.dataset.export_config import DataExportType


@dataclass
class ExportInfo:

    export_paths: Dict[str, str]

    def __post_init__(self):
        allowed_keys = tuple(a.value for a in DataExportType)
        for key in self.export_paths.keys():
            if key not in allowed_keys:
                raise TypeError(
                    f"The allowed keys for export type are: {allowed_keys}. Got {key} "
                    f"which is not in the allowed list"
                )

    def to_str(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_str(cls, string: str) -> "ExportInfo":
        datadict = json.loads(string)
        return cls(**datadict)
