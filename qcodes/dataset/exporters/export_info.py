"""
This module defines the ExportInfo dataclass
"""
import json
from dataclasses import asdict, dataclass
from typing import Dict

from qcodes.dataset.export_config import DataExportType


@dataclass
class ExportInfo:

    export_paths = Dict[DataExportType, str]

    def to_str(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_str(cls, string: str) -> "ExportInfo":
        datadict = json.loads(string)
        return cls(**datadict)
