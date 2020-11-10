from typing import Tuple
from itertools import product

import numpy as np

from qcodes.plots.qcmatplotlib import MatPlot


def make_simulated_xyz(xrange: np.ndarray,
                       yrange: np.ndarray,
                       step: int,
                       interrupt_at: int=0) -> Tuple[np.ndarray,
                                                     np.ndarray,
                                                     np.ndarray]:
    """
    Make x, y, and z np.arrays like a Loop measurement would. In particular
    get the positions of nans right.

    Args:
        xrange: a 1D array with inner loop set values
        yrange: a 1D array with outer loop set values
        step: is the outer loop (y) step, describes how far we are in the
          measurement (0: all NaNs, len(y): all data)
        interrupt_at: inner loop step to interrupt at

    Returns:
        (x, y, z) where z is random noise
    """
    y = yrange.copy()
    y[step:] = np.nan

    x = np.empty((len(yrange), len(xrange)))
    x.fill(np.nan)
    z = x.copy()
    for stepind in range(step):
        xrow =  xrange.copy()
        if stepind == step - 1:
            xrow[interrupt_at:] = np.nan
        zrow = np.random.randn(len(xrange))
        if stepind == step - 1:
            zrow[interrupt_at:] = np.nan
        x[stepind, :] = xrow
        z[stepind, :] = zrow

    return x, y, z


def test_make_args_for_pcolormesh():
    # We test some common situations
    #
    # y in the outer loop setpoints, i.e. of shape (N,)
    # x is the inner loop setpoints, i.e. of shape (N, M)
    # z is the data, i.e. of shape (N, M)

    N = 10  # y
    M = 25  # x

    xrange = np.linspace(-1, 1, M)
    yrange = np.linspace(-10, 0.5, N)

    # up scans, down scans

    for xsign, ysign in product([-1, 1], repeat=2):

        x, y, z = make_simulated_xyz(xsign*xrange,
                                     ysign*yrange,
                                     step=N//2+1)

        args_masked = [np.ma.masked_invalid(arg) for arg in [x, y, z]]

        args = MatPlot._make_args_for_pcolormesh(args_masked, x, y)

        assert len(args[0]) == M + 1
        assert len(args[1]) == N + 1

    # an interrupted scan

    x, y, z = make_simulated_xyz(xsign*xrange,
                                 ysign*yrange,
                                 step=N//2+1,
                                 interrupt_at=M//2+1)

    args_masked = [np.ma.masked_invalid(arg) for arg in [x, y, z]]

    args = MatPlot._make_args_for_pcolormesh(args_masked, x, y)

    assert len(args[0]) == M + 1
    assert len(args[1]) == N + 1
