import numpy as np
from scipy.interpolate import griddata


def interpolate2D(x, y, z, nx=None, ny=None, method='nearest',
                  fill_value=np.nan):
    '''
    Interpolate `x,y,z` data on a 2D grid.
    Based on ``scipy.interpolate.griddata``

    Args:
        x (1D or 2D array): Array that defines datapoints along the x-axis.
            If x is 1D it is assumed to be the same for every step in y.

        y (1D or 2D array): Array that defines datapoints along the y-axis.
            If y is 1D it is assumed to be the same for every step in x.

        z (1D or 2D array): Z value array.

        nx (Optional[int]): Optional Number of points to interpolate along
            x-axis. Default `None`.

        ny (Optional[int]): Optional Number of points to interpolate along
            y-axis. Default `None`.

        method (Optional[str]): Method of interpolation, one of
            'linear', 'nearest', 'cubic'.   Default  'nearest'.

        fill_value (float) Passed directly to
            ``scipy.interpolate.griddata``.

    Returns:
         Tuple(numpy.array, numpy.array, numpy.array): x-values with len(n),
            y-values with len(m), z-values with shape(n,m) where n and m are
            the dimension of the interpolated grid.
    '''

    if nx is not None:
        xvals = np.linspace(np.nanmin(x), np.nanmax(x), nx)
    else:
        xvals = np.unique(x)
        xvals = xvals[np.isfinite(xvals)]
        nx = xvals.size

    if ny is not None:
        yvals = np.linspace(np.nanmin(y), np.nanmax(y), ny)
    else:
        yvals = np.unique(y)
        yvals = yvals[np.isfinite(yvals)]
        ny = yvals.size

    xs, ys = x.shape, y.shape
    if xs != ys:
        if len(ys) < len(xs):
            if xs[0] == ys[0]:
                y = np.repeat([y], xs[1], 1)
        elif len(ys) > len(xs):
            if xs[0] == ys[0]:
                x = np.repeat([x], ys[1], 1)
    gridx, gridy = np.meshgrid(xvals, yvals)
    x = np.hstack(x)
    y = np.hstack(y)
    z = np.hstack(z)
    points = np.stack((x, y)).T
    values = griddata(points, z, (gridx, gridy), method=method, fill_value=fill_value)

    return gridx[1, :], gridy[:, 1], values
