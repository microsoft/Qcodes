"""
Legacy module kept around for backwards compatibility reasons.
Will eventually be deprecated and removed

"""

from .BaselSP983c import BaselSP983c


class SP983C(BaselSP983c):
    def get_idn(self) -> dict[str, str | None]:
        vendor = "Physics Basel"
        model = "SP 983(c)"
        serial = None
        firmware = None
        return {
            "vendor": vendor,
            "model": model,
            "serial": serial,
            "firmware": firmware,
        }
