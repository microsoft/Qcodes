import numpy as np
from qcodes.dataset.param_spec import ParamSpec

def dataset_with_outliers_generator(ds, data_offset=5, low_outlier=-3,
                 high_outlier=1, background_noise=True):
    x = ParamSpec('x', 'numeric', label='Flux', unit='e^2/hbar')
    t = ParamSpec('t', 'numeric', label='Time', unit='s')
    z = ParamSpec('z', 'numeric', label='Majorana number', unit='Anyon',
                depends_on=[x, t])
    ds.add_parameter(x)
    ds.add_parameter(t)
    ds.add_parameter(z)
    ds.mark_started()

    npoints = 50
    xvals = np.linspace(0, 1, npoints)
    tvals = np.linspace(0, 1, npoints)
    for counter, xv in enumerate(xvals):
        if background_noise and (
            counter < round(npoints/2.3) or counter > round(npoints/1.8)):
            data = np.random.rand(npoints)-data_offset
        else:
            data = xv * np.linspace(0,1,npoints)
        if counter == round(npoints/1.9):
            data[round(npoints/1.9)] = high_outlier
        if counter == round(npoints/2.1):
            data[round(npoints/2.5)] = low_outlier
        ds.add_results([{'x': xv, 't': tv, 'z': z}
                            for z, tv in zip(data, tvals)])
    ds.mark_completed()
    return ds
