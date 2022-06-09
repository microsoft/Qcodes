"""
The extensions module contains smaller modules that extend the functionality of QCoDeS.
These modules may import from all of QCoDeS but do not them self get imported into QCoDeS.
"""
from .installation import register_station_schema_with_vscode
from .slack import Slack, SlackTimeoutWarning

__all__ = ["Slack", "SlackTimeoutWarning", "register_station_schema_with_vscode"]
