import qcodes as qc
import numpy as np
import types
import os
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
from scipy.optimize import curve_fit
from qcodes import config

from qcodes.dataset.data_set import load_by_id
from qdev_wrappers.dataset.plotting import plot_by_id
from qcodes.dataset.data_export import get_data_by_id



class qdev_fitter():
    def __init__(self):
        self.T1 = T1()
        self.T2 = T2()

    def fit(self, dataid, fitclass, save_plots=True, p0=None,**kwargs):

        ax_list, _ = plot_by_id(dataid)
        popt_list = []
        pcov_list = []
        for i, ax in enumerate(ax_list):
            if ax.lines == []:
                print(f'No line found in plot {i}.')
            else:
                xdata = ax.lines[0].get_xdata()
                ydata = ax.lines[0].get_ydata()
                # Get initial guess on parameter is guess function is defined
                if (p0 is None and hasattr(fitclass,'guess')):
                    p0 = getattr(fitclass,'guess')(xdata, ydata)
                popt, pcov = curve_fit(fitclass.fun, xdata, ydata, p0=p0, **kwargs)
                popt_list.append(popt)
                pcov_list.append(pcov)

                if save_plots:
                    self.plot_1D(ax, xdata, ydata, fitclass, popt)

                    dataset = load_by_id(dataid)
                    mainfolder = config.user.mainfolder
                    experiment_name = dataset.exp_name
                    sample_name = dataset.sample_name

                    storage_dir = os.path.join(mainfolder, experiment_name, sample_name)
                    analysis_dir = os.path.join(storage_dir, 'Analysis')
                    os.makedirs(analysis_dir, exist_ok=True)

                    full_path = os.path.join(analysis_dir, f'{dataid}_{i}.png')
                    ax.figure.savefig(full_path, dpi=500)
        return popt_list, pcov_list


    def plot_1D(self, ax, xdata, ydata, fitclass, popt):
        ax.lines[0].set_linestyle('')
        ax.lines[0].set_marker('.')
        ax.lines[0].set_markersize(5)
        ax.lines[0].set_color('C0')
        ax.figure.set_size_inches(6.5,4)
        ax.figure.tight_layout(pad=3)

        # Get labels for fit results with correct scaling
        p_label_list = []
        for i in range(len(fitclass.p_names)):
            ax_letter = fitclass.p_units[i]
            if ax_letter in ['x','y']:
                unit = getattr(ax, 'get_{}label'.format(ax_letter))().split('(')[1].split(')')[0]
                scaled = float(getattr(ax, '{}axis'.format(ax_letter)).get_major_formatter()(popt[i]).replace('−','-'))
            elif ax_letter in ['1/x','1/y']:
                unit = '/{}'.format(getattr(ax, 'get_{}label'.format(ax_letter[2]))().split('(')[1].split(')')[0])
                scaled = 1/float(getattr(ax, '{}axis'.format(ax_letter[2])).get_major_formatter()(1/popt[i]).replace('−','-'))
            else:
                unit = ax_letter
                scaled = popt[i]
            p_label_list.append('{} = {:.3g} {}'.format(fitclass.p_names[i],scaled,unit))
        x = np.linspace(xdata.min(),xdata.max(),len(xdata)*10)
        ax.plot(x,fitclass.fun(x,*popt),color='C0')
        ax.figure.text(0.8, 0.45, '\n'.join(p_label_list),bbox={'ec':'k','fc':'w'})
        ax.set_title(fitclass.fun_str)
        ax.figure.subplots_adjust(right=0.78)


#  Predefined fucntions
class T1():
    def __init__(self):
        self.name = 'T1fit'
        self.fun_str = r'$f(x) = a \exp(-x/T) + c$'
        self.p_names = [r'$a$',r'$T$',r'$c$']
        self.p_units = ['y','x','y']

    def fun(self,x,a,T,c):
        val = a*np.exp(-x/T)+c
        return val

    def guess(self,x,y):
        l = len(y)
        val_init = y[0:round(l/20)].mean()
        val_fin = y[-round(l/20):].mean()
        a = val_init - val_fin
        c = val_fin
        # guess T1 as point where data has falen to 1/e of init value
        idx = (np.abs(y-a/np.e-c)).argmin()
        T = x[idx]
        return [a,T,c]


class T2():
    def __init__(self):
        self.name = 'T2fit'
        self.fun_str = r'$f(x) = a \sin(\omega x +\phi)\exp(-x/T) + c$'
        self.p_names = [r'$a$',r'$T$',r'$\omega$',r'$\phi$',r'$c$']
        self.p_units = ['y','x','1/x','','y']

    def fun(self,x,a,T,w,p,c):
        val = a*np.exp(-x/T)*np.sin(w*x+p)+c
        return val

    def guess(self,x,y):
        a = y.max() - y.min()
        c = y.mean()
        # guess T2 as point half way point in data
        T = x[round(len(x)/2)]
        # Get initial guess for frequency from a fourier transform
        yhat = fftpack.rfft(y-y.mean())
        idx = (yhat**2).argmax()
        freqs = fftpack.rfftfreq(len(x), d = (x[1]-x[0])/(2*np.pi))
        w = freqs[idx]
        p = 0
        return [a,T,w,p,c]