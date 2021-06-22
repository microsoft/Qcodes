"""
This module defines the ExportInfo dataclass
"""
import enum
import json
from dataclasses import asdict, dataclass
from typing import Dict

from qcodes.dataset.export_config import DataExportType


@dataclass
class ExportInfo:

    export_paths: Dict[DataExportType, str]

    def to_str(self) -> str:
        return json.dumps(asdict(self), cls=HandleKeyEnumEncoder)

    @classmethod
    def from_str(cls, string: str) -> "ExportInfo":
        datadict = json.loads(string)
        return cls(**datadict)


def _sanitize(o):
    if isinstance(o, enum.Enum):
        return o.value
    return o


class HandleKeyEnumEncoder(json.JSONEncoder):
    def encode(self, o):
        return super().encode(_sanitize(o))
