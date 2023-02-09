from qcodes.utils import issue_deprecation_warning

try:
    from qcodes_loop.extensions.slack import Slack, SlackTimeoutWarning, convert_command
except ImportError as e:
    raise ImportError(
        "qcodes.utils.slack is deprecated and has moved to "
        "the package `qcodes_loop`. Please install qcodes_loop directly or "
        "with `pip install qcodes[loop]"
    ) from e



__all__ = ["Slack", "SlackTimeoutWarning", "convert_command"]
issue_deprecation_warning(
    "qcodes.utils.slack module", alternative="qcodes_loop.extensions.slack"
)
