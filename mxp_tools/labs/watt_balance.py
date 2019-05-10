import os
import csv
import pandas
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import OrderedDict
import scipy.stats
from scipy.signal import savgol_filter

from mxp_tools.fit import Analysis, Value, Model
# pylint: disable=bad-whitespace


class WattBalance:
    pass


class ForceMode:

    def __init__(self, data_file):
        self.g = Value(9.8058, 2/1e5)

        self.raw_data = self._load_raw(data_file)

        self.analysis = None
        self.I = None
        self.mg = None

    def process(self):
        raw_data = self.raw_data
        self.I = self._get_I(raw_data)
        self.mg = self._get_mg(raw_data, self.g)

        x = self.mg.value
        y = self.I.value
        yerr = self.I.err

        self.analysis = self._analysis(x, y, yerr)

    @property
    def xye(self):
        return self.I, self.mg.value, self.mg.err

    @staticmethod
    def _load_raw(data_file):
        return pandas.read_csv(data_file)

    def load_data(self):
        data = self.raw_data

    @staticmethod
    def _get_I(raw_data):
        I1 = np.mean(raw_data[['I1', 'I3']], axis=1)
        I2 = np.mean(raw_data[['I2', 'I4']], axis=1)

        I1err = np.std(raw_data[['I1', 'I3']], ddof=1, axis=1)
        I2err = np.std(raw_data[['I2', 'I4']], ddof=1, axis=1)

        I1err = np.max([I1err, np.ones_like(I1)*0.01], axis=0)
        I2err = np.max([I2err, np.ones_like(I2)*0.01], axis=0)
        # I1err = np.sqrt(np.sum(raw_data[['I1 err', 'I3 err']], axis=1))
        # I2err = np.sqrt(np.sum(raw_data[['I2 err', 'I4 err']], axis=1))

        I = I2 - I1
        Ierr = np.sqrt(I1err**2+I2err**2)
        return Value(I, Ierr)

#     @staticmethod
#     def _get_I(raw_data):
#         I1 = raw_data['I1']
#         I2 = raw_data['I2']
# 
#         I1err = raw_data['I1 err']
#         I2err = raw_data['I2 err']
# 
#         I = I1 + I2
#         Ierr = np.sqrt(I1err**2 + I2err**2)
# 
#         return Value(I, Ierr)

    @staticmethod
    def _get_m(raw_data):
        return Value(raw_data['mass'], 0.0001)

    @classmethod
    def _get_mg(cls, raw_data, g):
        m = cls._get_m(raw_data)
        m, merr = m.value, m.err
        g, gerr = g.value, g.err

        mg = m*g
        mgerr = np.sqrt((m*gerr)**2+(g*merr)**2)

        return Value(mg, mgerr)

    @staticmethod
    def _analysis(x, y, yerr):
        analysis = Analysis.new_2param(x, y, yerr)
        return analysis

    def get_slope(self):
        v = self.analysis.model.params[0]
        e = self.analysis.model.param_errs[0]
        return Value(v, e)

    def get_BL(self):
        slope = self.get_slope()
        bl = 1/slope.value
        blerr = slope.value/bl**2
        return Value(bl, blerr)

    def plot(self, fig):
        fig.suptitle('Force')
        ax = fig.add_subplot(121)
        self.analysis.plot(ax)

        ax.set_xlabel('')

        ax = fig.add_subplot(122)
        self.analysis.plot_chi(ax)
        ax.grid()

    def __str__(self):
        return 'Sample\n' \
               '--------\n' \
               'Analysis\n' \
               '--------\n' \
               '{}\n\n' \
               'slope: {}\n' \
               'BL: {}\n' \
               .format(self.analysis, self.get_slope(), self.get_BL())

    def __repr__(self):
        return str(self)


class VelocityModeData:

    def __init__(self, data_file, data_range=None):
        self.fname = data_file
        self.raw_data = self._load_data(data_file)

        self.data_range = data_range or [1000, -1000]
        self.savgol_window = 51
        self.savgol_poly = 3
        self.use_savgol = True

        self.stage1 = None
        self.stage2 = None
        self.corr_data = None

        self.mean_position = None
        self.shift = None
        self.analysis = None

    def process(self):
        xy, _ = self.load_data()
        self.mean_position = self._data_mean_pos(xy)

        dx = self._velocity_savgol(xy[:,0])
        xy = np.array([dx, xy[:,1]]).T

        xy = self._data_chop(xy)
        xy = self._data_offset(xy, self.data_range)
        self.stage1 = xy.copy()

        shift, corr = self._data_correlation(xy)
        # print('shift: {}'.format(shift))
        self.shift = shift
        self.corr_data = corr

        self.stage2 = self._data_shift(xy, shift)
        self.analysis = self._analysis(self.stage2)
        # self.slope = self.regress(self.stage2)

    @property
    def stage1_norm(self):
        return self._data_normalize(self.stage1)

    @property
    def stage2_norm(self):
        return self._data_normalize(self.stage2)

    @staticmethod
    def _load_data(data_file):
        return pandas.read_csv(data_file, header=-1)

    def load_data(self):
        data = self.raw_data
        position = data[0].as_matrix()
        voltage = data[2].as_matrix()

        (position_t, position_i), (_, voltage_i) = \
            self._data_match_time(data)

        position = position[position_i]
        # print(position)
        position /= 1000
        # print(position)
        voltage = voltage[voltage_i]

        return np.array([position, voltage]).T, np.array(position_t)[position_i]

    def _data_match_time(self, raw_data):
        x_raw = raw_data[1].as_matrix()
        y_raw = raw_data[3].as_matrix()
        parse_str = '%Y%m%d-%H:%M:%S.%f'

        def parse(s):
            return datetime.strptime(s, parse_str)

        def get_dt(t1, t2):
            return (t1-t2).total_seconds()

        x = []
        y = []
        for i in np.arange(x_raw.shape[0]):
            if type(x_raw[i]) is str:
                x.append(parse(x_raw[i]))

            if type(y_raw[i]) is str:
                y.append(parse(y_raw[i]))

        out = []
        for i in np.arange(len(x)):
            for j in np.arange(max(i-5,0), len(y)):
                dt = get_dt(x[i], y[j])
                if dt < 0.005:
                    out.append((i, j))
                    break

        xi, yi = zip(*out)
        xi, yi = np.array(xi), np.array(yi)

        x = [x[i] for i in xi]
        y = [y[i] for i in yi]
        return (x, xi), (y, yi)

    @staticmethod
    def _data_chop(xy):
        x = xy[:,1]
        i = np.where(np.abs(x)<0.3)[0]

        return xy[i,:]

    @staticmethod
    def _data_normalize(xy):
        return (xy - xy.mean(axis=0))/xy.std(axis=0)

    @staticmethod
    def _data_offset(xy, data_range):
        return xy[data_range[0]:data_range[1]]

    @staticmethod
    def _data_shift(xy, n):
        if n>0:
            x = xy[n:,0]
            y = xy[:-n,1]
        elif n<0:
            n = -n
            x = xy[:-n,0]
            y = xy[n:,1]
        else:
            return xy.copy()
        return np.array([x, y]).T

    def _velocity_savgol(self, x):
        # print(x)
        return self._data_savgol(x, deriv=1)

    def _data_savgol(self, x, deriv=1):
        return savgol_filter(
            x,
            window_length=self.savgol_window,
            polyorder=self.savgol_poly,
            deriv=deriv,
            delta=0.01)

    def _velocity_diff(self, x, xt):
        # x = xy[:,0]
        dx = np.zeros_like(x)
        for i in np.arange(1, x.shape[0]-1):
            dt = abs((xt[i+1]-xt[i-1]).total_seconds())
            dx[i] = (x[i+1]-x[i-1])/dt

        # dx = dx[1:-1]/1000 # mm/s->m/s
        return dx
        dx = dx[1:-1]
        return np.array([dx, xy[1:-1,1].copy()]).T

    @staticmethod
    def _data_mean_pos(xy):
        x = xy[-300:-100,0]
        return np.mean(x)

    def _data_correlation(self, xy):
        xy = self._data_normalize(xy)

        corr = np.correlate(xy[:,0], xy[:,1], 'full')
        N = xy.shape[0]
        i = np.arange(-N+1, N)
        # self.data['corr'] = (i, corr)
        max_corr = i[np.argmax(corr)]
        return max_corr, (i, corr)

    def _uncertainty(self, x):
        window = self.savgol_window
        xe = np.zeros_like(x)
        for i in np.arange(window, x.shape[0] - window):
            xe[i] = np.std(x[i-window:i+window], ddof=1)
        return np.mean(xe[window:-window])*np.ones_like(x)

    def fit_chi2(self, xy):
        y = xy[:,0]
        x = xy[:,1]
        yerr = np.ones_like(y)*0.01*1e-3
        def fit(x, a, b):
            return a*x+b

        popt, pcov = scipy.optimize.curve_fit(fit, x, y, sigma=yerr)
        perr = np.sqrt(np.diag(pcov))
        model = Model(fit, popt, perr)
        analysis = Analysis(model, x, y, yerr)

        return analysis

    def regress(self, xy):
        return scipy.stats.linregress(xy[:,0], xy[:,1])

    def _analysis(self, xy):
        x, y = xy[:,0], xy[:,1]
        yerr = self._uncertainty(y)
        analysis = Analysis.new_2param(x, y, yerr)
        return analysis

    def get_slope(self):
        v = self.analysis.model.params[0]
        e = self.analysis.model.param_errs[0]
        return Value(v, e)


class VelocityMode:

    def __init__(self, data_file, data_range=None):
        self.data = VelocityModeData(data_file, data_range=data_range)
        self.dt = 10/1000 # 10ms

        self.max_corr = None

        self.data_cleaned = None
        self.data_shifted = None

    def summary(self):
        data = {
            'shift': self.data.shift,
            'mean_pos': self.data.mean_position*1000,
            'slope': self.data.get_slope()
        }
        return data

    def scatter(self, xy, ax, title=None):
        x, y = xy[:,0], xy[:,1]
        ax.scatter(x, y, s=1, alpha=.8)

        ax.set_xlabel('Velocity (m/s)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(title)

    @staticmethod
    def plot_time_series(xy, ax, offset=500, legend_loc=None):
        i = np.arange(min(xy.shape[0], offset))/100
        ax.plot(i, xy[:offset,0], alpha=1, label='Velocity')
        ax.plot(i, xy[:offset,1], alpha=0.5, label='Voltage')

        ax.set_xlabel('time (s)')
        ax.legend(loc=legend_loc)

    def plot_xy_with_regression(self, xy, ax, slopes=None):
        self.data.analysis.plot(ax, plot_ebars=False)
#         self.scatter(xy, ax)
#         
#         result = self.data.regress(xy)
#         print(result)
#         # result = self.data.slope
#         x = np.linspace(min(xy[:,0]), max(xy[:,0]), 100)
#         y = result.slope*x+result.intercept
#         ax.plot([],[])
#         ax.plot(x, y)
# 
#         if slopes:
#             for s in slopes:
#                 y = s*x+result.intercept
#                 ax.plot(x, y)

    def process(self):
        self.data.process()

        xy = self.data.stage1_norm
        fig = plt.figure()
        ax = fig.add_subplot(111)
        i = np.arange(xy.shape[0])
        ax.plot(i, xy[:,0])
        ax.plot(i, xy[:,1])
        plt.show()


        fig = plt.figure(figsize=(15,10))
        fig.suptitle(self.data.fname.split('/')[-1])

        xy = self.data.stage1
        ax = fig.add_subplot(221)
        self.scatter(xy, ax, title='Before Shifting')

        ax = fig.add_subplot(222)
        self.plot_time_series(self.data.stage1_norm, ax)

        xy = self.data.stage2
        ax = fig.add_subplot(223)
        self.scatter(xy, ax, title='After Shifting')

        ax = fig.add_subplot(224)
        self.plot_time_series(self.data.stage2_norm, ax)

        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.plot_xy_with_regression(xy, ax)
        # self.scatter(xy, ax)
        
        # result = self.data.slope
        # x = np.linspace(min(xy[:,0]), max(xy[:,0]), 100)
        # y = result.slope*x+result.intercept
        # ax.plot(x, y)

        plt.show()
        # print(self.data.slope)
