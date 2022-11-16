import warnings

from qcodes.extensions.slack import Slack, SlackTimeoutWarning, convert_command

# todo enable warning once new api is in release
# warnings.warn(
#     "qcodes.utils.slack module is deprecated. "
#     "Please update to import from qcodes.extensions"
# )

__all__ = ["Slack", "SlackTimeoutWarning", "convert_command"]
