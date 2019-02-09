import numpy as np

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']


class Analysis:

    def __init__(self, model, x, y, yerr):
        self.model = model
        self.x = x
        self.y = y
        self.yerr = yerr

        self.kai = self._kai(x, y, yerr, model)
        self.kai2 = np.sum(self.kai**2)
        self.kai2_red = self.kai2 / (len(x) - model.n_params)

    def __str__(self):
        return '{}\nReduced chi^2: {:.2f}'.format(self.model, self.kai2_red)

    def plot(self, ax, ebar_args=None):
        kwargs = {
            'fmt': '.',
            'capsize': 5,
            'color': COLORS[0]
        }
        if ebar_args:
            kwargs.update(ebar_args)

        ax.errorbar(self.x, self.y, self.yerr, **kwargs)

        x = np.linspace(np.min(self.x), np.max(self.x), 100)
        y_min, y_max = self.model.bounds(x)
        y = self.model(x)
        ax.plot(x, y, color=COLORS[0], alpha=0.8)

        fmt = {
            'alpha': 1,
            'color': COLORS[3],
            'linewidth': 0.7
        }
        ax.plot(x, y_min, '--', **fmt)
        ax.plot(x, y_max, '--', **fmt)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(['Data', 'Fit', 'Bounds'])

        return ax

    def plot_chi(self, ax):
        ax.plot(self.x, self.kai, '.')

        ax.set_xlabel('x')
        ax.set_ylabel('chi')
        ax.set_title('Weighted Residuals')
        
        return ax

    @staticmethod
    def _kai(x, y, yerr, fx):
        return ((y-fx(x))/yerr)

    @staticmethod
    def lsq_fit(x, y, ey):
        #y=a*x+b
        sx = np.sum(x / ey**2)
        sy = np.sum(y / ey**2)
        sxx = np.sum(x**2 / ey**2)
        sxy = np.sum(x*y / ey**2)
        s = np.sum(1/ey**2)

        Delta = sxx*s-sx*sx

        a = (s*sxy - sx*sy) / Delta;
        ea = np.sqrt(s / Delta);
        b = (sxx*sy - sx*sxy) / Delta;
        eb = np.sqrt(sxx / Delta);

        return a, ea, b, eb


class Model:

    def __init__(self, function, params, param_errs):
        self.function = function
        self.params = params
        self.param_errs = param_errs

    @property
    def n_params(self):
        return len(self.params)

    def __str__(self):
        return 'params {} param errors {}'.format(self.params, self.param_errs)

    def __call__(self, x):
        return self.function(x, *self.params)

    def bounds(self, x):
        params = self.params
        param_errs = self.param_errs

        new_params = np.zeros((2**self.n_params, self.n_params),
                              dtype=np.float32)
        a = True
        b = True
        for i in np.arange(new_params.shape[0]):
            ii = i
            for j in np.arange(new_params.shape[1]):
                p = ii % 2
                ii = ii // 2

                new_params[i,j] = p

        new_params[np.where(new_params == True)] = 1
        new_params[np.where(new_params == False)] = -1

        for i in np.arange(new_params.shape[0]):
            new_params[i,:] = params + new_params[i,:] * param_errs

        y = np.zeros((x.shape[0], new_params.shape[0]))
        for i in np.arange(new_params.shape[0]):
            y[:,i] = self.function(x, *new_params[i,:])

        y_min = np.min(y, axis=1)
        y_max = np.max(y, axis=1)

        return y_min, y_max

class FitLSQ:
    @staticmethod
    def lsq_fit(x, y, a):
        # y = mx+c
        # w = 1/a^2
        w = 1/a**2
        Delta = np.sum(w)*np.sum(w*x**2) - np.sum(w*x)**2

        c = (np.sum(w*x**2)*np.sum(w*y) - np.sum(w*x)*np.sum(w*x*y))/Delta
        m = (np.sum(w)*np.sum(w*x*y) - np.sum(w*x)*np.sum(w*y))/Delta

        c_err = np.sqrt(np.sum(w*x**2)/Delta)
        m_err = np.sqrt(np.sum(w)/Delta)

        return c, m, c_err, m_err


    @staticmethod
    def lsq_fit2(x, y, ey):
        #y=a+b*x
        sx = np.sum(x / ey**2)
        sy = np.sum(y / ey**2)
        sxx = np.sum(x**2 / ey**2)
        sxy = np.sum(x*y / ey**2)
        s = np.sum(1/ey**2)

        Delta = sxx*s-sx*sx

        a = (sxx*sy - sx*sxy) / Delta;
        ea = np.sqrt(sxx / Delta);
        b = (s*sxy - sx*sy) / Delta;
        eb = np.sqrt(s / Delta);

        return a, ea, b, eb


    @staticmethod
    def fx(c, m):
        def f(x):
            return m*x+c
        return f

    @staticmethod
    def Kai(x, y, yerr, fx):
        return ((y-fx(x))/yerr)

    @staticmethod
    def find_intersect(fx, value, guess, precision=1e-6):
        step = 1

        x = guess
        while 1:
            y = fx(x)
            y2 = fx(x+step)

            if y2-y < precision:
                break


            dy = (fx(x-step/2)-fx(x+step/2))/step
            print(x, y, y2, dy)

            x = (y2-y)/dy + x





