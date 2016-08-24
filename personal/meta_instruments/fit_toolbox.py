import numpy as np
import lmfit

class Fit():
    def __init__(self):
        self.model = lmfit.Model(self.fit_function)

    def fit_function(self, ydata, xvals):
        pass

    def find_initial_parameters(self):
        pass

    def perform_fit (self):
        pass

    def find_nearest(self, array, value):
        idx = (np.abs(array - value)).argmin()
        return array[idx]

class ExponentialFit(Fit):
    def __init__(self):
        super().__init__()

    def fit_function(self, t,  tau, amplitude, offset):
        super().__init__()
        return amplitude * np.exp(t/tau) + offset

    def find_initial_parameters(self, xvals, ydata, initial_parameters):
        super().__init__()
        if not 'tau' in initial_parameters:
            initial_parameters['tau'] = xvals[1] - xvals[np.where(ydata == self.find_nearest(ydata, ydata[1] / np.exp(1)))]
        if not 'amplitude' in initial_parameters:
            initial_parameters['amplitude'] = ydata[1]
        if not 'offset' in initial_parameters:
            initial_parameters['offset'] = ydata[-1]

        print('iniiiiiiit')
        print(initial_parameters)

        self.model.make_params(tau=initial_parameters['tau'],
                               amplitude=initial_parameters['amplitude'],
                               offset=initial_parameters['offset'])


    def perform_fit(self, xvals, ydata, initial_parameters=None, options=None):
        super().__init__()

        if initial_parameters is None:
            initial_parameters={}

        self.find_initial_parameters(xvals, ydata, initial_parameters)

        print(ydata, xvals)


        return self.model.fit(ydata, t=xvals)
