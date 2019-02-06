from mxp_tools.fit import Analysis, FitLSQ, Model
import numpy as np

x = np.array([0.01, 1.1, 1.9])
y = np.array([2.3, 4.4, 6.7])
yerr = np.array([0.1, 0.1, 0.1])*2

b, m, b_err, m_err = FitLSQ.lsq_fit(x, y, yerr)

def f(x, m, b):
    return m*x+b

model = Model(f, (m, b), (m_err, b_err))
analysis = Analysis(model, x, y, yerr)

print(model.bounds(x))
print(model.params)
print(model.param_errs)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
analysis.plot(ax)
plt.show()
