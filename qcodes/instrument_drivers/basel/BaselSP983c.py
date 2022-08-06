from typing import Dict, Optional

from .BaselSP983 import BaselSP983


class BaselSP983c(BaselSP983):
    def get_idn(self) -> Dict[str, Optional[str]]:
        vendor = "Physics Basel"
        model = "SP 983c"
        serial = None
        firmware = None
        return {
            "vendor": vendor,
            "model": model,
            "serial": serial,
            "firmware": firmware,
        }
