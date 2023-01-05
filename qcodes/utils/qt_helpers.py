from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QMainWindow


def foreground_qt_window(window: "QMainWindow") -> None:
    """
    Try as hard as possible to bring a qt window to the front. This
    will use pywin32 if installed and running on windows as this
    seems to be the only reliable way to foreground a window. The
    build-in qt functions often doesn't work. Note that to use this
    with pyqtgraphs remote process you should use the ref in that module
    as in the example below.

    Args:
        window: Handle to qt window to foreground.
    Examples:
        >>> Qtplot.qt_helpers.foreground_qt_window(plot.win)
    """
    try:
        import win32con
        from win32gui import SetWindowPos

        # use the idea from
        # https://stackoverflow.com/questions/12118939/how-to-make-a-pyqt4-window-jump-to-the-front
        SetWindowPos(
            window.winId(),  # pyright: ignore[reportGeneralTypeIssues]
            win32con.HWND_TOPMOST,  # = always on top. only reliable way to bring it to the front on windows
            0,
            0,
            0,
            0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW,
        )
        SetWindowPos(
            window.winId(),  # pyright: ignore[reportGeneralTypeIssues]
            win32con.HWND_NOTOPMOST,  # disable the always on top, but leave window at its top position
            0,
            0,
            0,
            0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW,
        )
    except ImportError:
        pass
    window.show()
    window.raise_()
    window.activateWindow()
