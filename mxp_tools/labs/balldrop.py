import numpy as np
import pandas

class BallDropLab:

    def __init__(self, d_height):
        data = self.load_data()
        self.data = data
        self.height = data['Height'], d_height
        self.nylon = data['Nylon Mean'], data['Nylon Uncertainty']
        self.steel = data['Steel Mean'], data['Steel Uncertainty']

        self.m_steel = 28.17
        self.m_nylon = 4.09
        self.t_inf = self.calculate_tinf()

        self.data['T minf'] = self.t_inf[0]
        self.data['T minf err'] = self.t_inf[1]

    def calculate_tinf(self):
        t_steel, d_steel = self.steel 
        t_nylon, d_nylon = self.nylon
        m_steel = self.m_steel
        m_nylon = self.m_nylon
        
        t_minf = (t_steel*m_steel - t_nylon*m_nylon)/(m_steel-m_nylon)
        d_minf = 1/(m_steel-m_nylon) \
            * np.sqrt((m_steel*d_steel)**2 + (m_nylon*d_nylon)**2)

        return t_minf, d_minf
    
    def calculate_xy(self):
        t, d_t = self.t_inf
        height, d_height = self.height
        height, d_height = self.height
        x = np.array(t)
        y = np.array((height)/t)
        y_err = np.array(np.sqrt(
            d_t**2 * ((height)/t**2)**2 + \
            d_height**2 * (1/t)**2))
        return x, y, y_err

    @classmethod
    def load_data(cls):
        raw_data = pandas.read_csv('ball_data.csv')
        calibration_data = pandas.read_csv('ball_calibration.csv')
# display(raw_data)

        data = pandas.DataFrame()
        data['Height'] = raw_data['Height']

        steel_names = ['Steel %d' % i for i in [1, 2, 3]]
        nylon_names = ['Nylon %d' % i for i in [1, 2, 3]]

        nylon_mean = np.mean(raw_data[nylon_names], axis=1)
        nylon_std = np.std(raw_data[nylon_names], axis=1)
        steel_mean = np.mean(raw_data[steel_names], axis=1)
        steel_std = np.std(raw_data[steel_names], axis=1)

        data['Nylon Mean'] = nylon_mean
        data['Nylon Stdev'] = nylon_std
        data['Steel Mean'] = steel_mean
        data['Steel Stdev'] = steel_std

        nylon_calib_stdev = np.std(calibration_data['nylon'])
        steel_calib_stdev = np.std(calibration_data['steel'])

        data['Nylon Uncertainty'] = np.maximum(
            nylon_std,
            np.ones_like(nylon_std)*nylon_calib_stdev)

        data['Steel Uncertainty'] = np.maximum(
            steel_std,
            np.ones_like(steel_std)*steel_calib_stdev)

        return data
