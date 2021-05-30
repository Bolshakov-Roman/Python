import numpy as mp 
import matplotlib.pyplot as plt
import pandas as pd
def read_data(input_file):
    input_data = np.loadtxt(input_file, delimiter = None)
    dates = pd.date_range('1950-01', periods = input_data.shape[0], freq = 'M')
    output = pd.Series(input_data[:, index], index = dates)
    return output
if __name__ =='__main__':
    input_file = "/ D / Documents / Мухаметзянов БФБО-01-20/ для практической 4 по Питону.txt"
    timeseries = read_data(input_file)
plt.figure()
timeseries.plot()
plt.show
timeseries['1980':'1990'].plot() #нарезка данных временного ряда
   <matplotlib.axes._subplots.AxesSubplot at 0xa0e4b00>
plt.show()
timeseries_mm = timeseries.resample("A").mean() #повторная выборка со средним
timeseries_mm.plot(style = 'g--')
plt.show()