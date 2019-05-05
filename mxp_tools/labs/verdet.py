
import numpy as np
import pandas
import pandas as pd
import os
import numpy as np
import scipy.integrate
import scipy.optimize

import matplotlib

from mxp_tools.fit import Analysis, Value, Model
matplotlib.rcParams.update({
    'font.size': 18, 'text.usetex': True})


class VerdetLab:

    def __init__(self, fname):
        self.fname = fname

        self.water = PhiSample(fname, 'water')
        self.air = PhiSample(fname, 'air')
        self.magnet = Magnet(fname)
        self.angle = AngleCorrection(fname)

        self.verdet = Verdet(self.water, self.air, self.magnet, self.angle)


class Verdet:
    
    def __init__(self, water, air, magnet, angle):
        self.water = water
        self.air = air
        self.magnet = magnet
        self.angle = angle

    def __str__(self):
        return 'Verdet constant\n' \
               '---------------\n' \
               'Slop: {}\n' \
               'verdet: {}'.format(self.get_slope(), self.get_verdet())

    def get_verdet(self):
        m_water = self.water.get_slope()
        m_air = self.air.get_slope()
        m = self.get_slope()


        gamma = self.magnet.get_gamma()
        k = self.angle.get_k()

        print(m_air, m_water, m, gamma, k)
        verdet = self._f_verdet(m, gamma, k)

        return verdet

    def get_slope(self):
        m_water = self.water.get_slope()
        m_air = self.air.get_slope()
        return self._f_slope(m_water, m_air)

    @staticmethod
    def _f_verdet(m, gamma, k):
        v = m.v * k.v / gamma.v
        e = [(m.e*k.v/gamma.v)**2,
             (gamma.e*m.v*k.v/gamma.v**2)**2,
             (k.e*m.v/gamma.v)**2]
        e = np.sqrt(sum(e))

        v = v * 180/np.pi*60
        e = e * 180/np.pi*60
        return Value(v, e)

    @staticmethod
    def _f_slope(m_water, m_air):
        v = m_water.v - m_air.v
        e = np.sqrt(m_water.e**2 + m_air.e**2)
        return Value(v, e)


class AngleCorrection:
    def __init__(self, fname):
        phi, va, dv = self.get_data(fname)
        self.phi = phi
        self.va = va
        self.dv = dv

        self.analysis = self._analysis()

    def __str__(self):
        return 'Angle Calibration\n' \
               '--------\n' \
               'Analysis\n' \
               '--------\n' \
               '{}\n\n' \
               'k: {}'.format(self.analysis, self.get_k())

    def get_k(self):
        m = np.abs(self.analysis.model.params[0])
        merr = self.analysis.model.param_errs[0]
        
        v = 1/m
        e = merr*1/m**2

        return Value(v, e)

    def _analysis(self):
        x = self.phi
        y = self._f_phi_measured(self.dv, self.va)
        y, yerr = y.v, y.e

        analysis = Analysis.new_2param(x, y, yerr)
        return analysis

    def _f_phi_measured(self, dv, va):
        v = 1/2 * dv.v/(2*va.v-dv.v)
        e = [1/4 * (dv.e*(2*va.v/(2*va.v-dv.v)**2))**2,
             1/4 * (va.e*(2*dv.v/(2*va.v-dv.v)**2))**2]

        return Value(v, np.sqrt(sum(e)))

    def plot(self, fig):
        fig.suptitle('Angle Calibration')
        ax = fig.add_subplot(121)
        self.analysis.plot(ax)

        ax = fig.add_subplot(122)
        self.analysis.plot_chi(ax)
        ax.grid()

    @classmethod
    def get_data(cls, fname):
        data = pd.read_excel(fname, 'angle_calibration')

        delta_V = data['delta V (uV)']*1e-6
        delta_V_err = np.ones_like(delta_V)*2*1e-6

        phi = data['dPhi Real (degree)/25']/25*np.pi/180

        V_A = data['V_a (mV)']*1e-3
        V_A_err = np.ones_like(V_A)*2e-6

        va = Value(V_A, V_A_err)
        dv = Value(delta_V, delta_V_err)

        return phi, va, dv


class Magnet:

    L = Value(9.4673, 0.0024, 'cm')
    delta_x = Value(0, .2, 'cm')

    def __init__(self, fname):
        x, y, yerr = self.get_data(fname)
        self.x = x
        self.y = y
        self.yerr = yerr

        self.analysis = self._analysis()

    def __str__(self):
        return 'Magnet:\n' \
               '--------\n' \
               'Analysis\n' \
               '--------\n' \
               '{}\n\n' \
               'gamma: {}'.format(self.analysis, self.get_gamma())

    def get_gamma(self):
        model = self.analysis.model
        a = Value(model.params[0], model.param_errs[0])
        b = Value(model.params[1], model.param_errs[1])
        c = Value(model.params[2], model.param_errs[2])

        print('a', a)
        print('b', b)
        print('c', c)

        alpha = self._f_alpha(self.L, self.delta_x)
        beta = self._f_beta(self.L, self.delta_x)

        print('alpha', alpha)
        print('beta', beta)

        gamma = self._f_gamma(a, b, c, alpha, beta)
        return gamma

    @staticmethod
    def _f_beta(L, delta_x):
        v = L.v/2+ delta_x.v
        e = np.sqrt((L.e**2/4 + delta_x.e**2))
        return Value(v, e)

    @staticmethod
    def _f_alpha(L, delta_x):
        v = -L.v/2+ delta_x.v
        e = np.sqrt((L.e**2/4 + delta_x.e**2))
        return Value(v, e)

    @staticmethod
    def _f_gamma(a, b, c, alpha, beta):
        e = [(a.e/3*(beta.v**3-alpha.v**3))**2,
             (b.e/2*(beta.v**2-alpha.v**2))**2,
             (c.e*(beta.v-alpha.v))**2,
             (alpha.e*(a.v*alpha.v**2+b.v*alpha.v+c.v))**2,
             (beta.e*(a.v*beta.v**2+b.v*beta.v+c.v))**2]
        e = np.sqrt(sum(e))

        def f(x):
            return a.v/3*x**3+b.v/2*x**2+c.v*x

        v = f(beta.v) - f(alpha.v)

        return Value(v, e)

    def plot(self, fig):
        fig.suptitle('Magnetic Field Calibration')
        ax = fig.add_subplot(121)
        self.analysis.plot(ax)

        ax = fig.add_subplot(122)
        self.analysis.plot_chi(ax)
        ax.grid()

    def _analysis(self):
        x, y, yerr = self.x, self.y, self.yerr

        def fit(x, a, b, c):
            return a*x**2 + b*x + c

        popt, pcov = scipy.optimize.curve_fit(fit, x, y, sigma=yerr)
        perr = np.sqrt(np.diag(pcov))
        model = Model(fit, popt, perr)
        analysis = Analysis(model, x, y, yerr)

        return analysis

    @classmethod
    def get_data(cls, fname):
        data = pd.read_excel(fname, 'coil_calibration')
        x = data['x (cm)']
        y = data['B (Gauss)']
        yerr = np.ones_like(y)*0.1

        i = np.where(np.abs(x)<5.25)[0]
        x = x[i]
        y = y[i]
        yerr = yerr[i]

        return x, y, yerr


class PhiSample:

    def __init__(self, fname, name):
        self.name = name
        I, phi = self._get_data(fname, name)
        self.I = I
        self.phi = phi
        self.phi_err = np.ones_like(phi) * self._get_calib(fname, name)

        self.analysis = self._analysis()

    def __str__(self):
        return 'Sample {}\n' \
               '--------\n' \
               'Analysis\n' \
               '--------\n' \
               '{}\n\n' \
               'slope: {}\n' \
               'err: {:.4e}' \
               .format(
                    self.name, self.analysis, self.get_slope(),
                    self.phi_err[0])

    def __repr__(self):
        return str(self)

    def _analysis(self):
        x = self.I
        y = self.phi
        yerr = self.phi_err

        analysis = Analysis.new_2param(x, y, yerr)
        return analysis

    def get_slope(self):
        v = self.analysis.model.params[0]
        e = self.analysis.model.param_errs[0]
        return Value(v, e)

    def plot(self, fig):
        fig.suptitle('Verdet ({})'.format(self.name))
        ax = fig.add_subplot(121)
        self.analysis.plot(ax)

        ax = fig.add_subplot(122)
        self.analysis.plot_chi(ax)
        ax.grid()

    @classmethod
    def _get_calib(cls, fname, name):
        name += '_calibration'
        data = pd.read_excel(fname, name)

        P = data['phi (rad)']
        # P_mean = np.mean(P)
        P_std = np.std(P, ddof=1)

        return P_std

    @classmethod
    def _get_data(cls, fname, name):
        name += '_data'
        data = pd.read_excel(fname, name)

        x = data['I (A)']
        y = data['phi (rad)']

        return x, y
