"""
Monitor a set of parameters in a background thread
stream output over websocket.

To start monitor, run this file, or if qcodes is installed as a module:

``% python -m qcodes.monitor.monitor``

Add parameters to monitor in your measurement by creating a new monitor with a
list of parameters to monitor:

``monitor = qcodes.Monitor(param1, param2, param3, ...)``
"""

from .monitor import Monitor

__all__ = ["Monitor"]
