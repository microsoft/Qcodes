import warnings

from qcodes.extensions.slack import Slack, SlackTimeoutWarning, convert_command

warnings.warn(
    "qcodes.utils.slack module is deprecated. "
    "Please update to import from qcodes.extensions"
)

__all__ = ["Slack", "SlackTimeoutWarning", "convert_command"]
