'''
import matplotlib.pyplot as plt
# %matplotlib inline # if we work in Jupyter notebook
plt.plot([1,2,3,4,5], [1,2,3,4,5])
plt.show()


import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,10,50)
y = x
plt.title("Линейная зависимость y = x")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.plot(x,y)
plt.show()



import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,10,50)
y1 = x
y2 = [i**2 for i in x]
plt.title("Зависимости: y1 = x, y2 = x^2")
plt.xlabel("x")
plt.ylabel("y1, y2")
plt.grid()
plt.plot(x,y1,x,y2)
plt.show()



import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,10,50)
y1= x
y2 = [i**2 for i in x]
plt.figure(figsize=(9,9))
plt.subplot(2,1,1)
plt.plot(x,y1)
plt.title("зависимости: y1 = x, y2 = x^2")
plt.ylabel("y1", fontsize=14)
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(x,y2)
plt.xlabel("x",fontsize=14)
plt.ylabel("y2",fontsize=14)
plt.grid(True)
plt.show()



import matplotlib.pyplot as plt
import numpy as np
fruits = ["apple", "peach","orange", "bannana", "melon"]
counts = [34,25,43,31,228]
plt.bar(fruits,counts)
plt.title("Fruins!")
plt.xlabel("Fruit")
plt.ylabel("Count")
plt.show()


import matplotlib.pyplot as plt
from matplotlib.ticker import(MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import numpy as np
x = np.linspace(0,10,10)
y1 = 4*x
y2 = [i**2 for i in x]
fig, ax = plt.subplots(figsize=(8,6))
ax.set_title("Grafuky zavisimostey: y1=4*x, y2=x^2", fontsize=16)
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y1,y2", fontsize=14)
ax.grid(which="major", linewidth=1.2)
ax.grid(which="minor", linestyle="--", color="grey",linewidth=0.5)
ax.scatter(x, y1, c="red", label="y1 = 4*x")
ax.plot (x, y2, label="y2 = x^2")
ax.legend()
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which = 'major', length=10, width=2)
ax.tick_params(which = 'minor', length=5, width=1)
plt.show()


import imp
import numpy as np
import sklearn.preprocessing
input_data = np.array([[2.1,-1.9,5.5], [-1.5,2.4,3.5], [0.5,-7.9,5.6], [5.9,2.3,-5.8]])
data_binarized = sklearn.preprocessing.Binarizer(threshold = 0.5).transform(input_data)
print("\nBinarized data: \n", data_binarized))


import imp
import numpy as np
import sklearn.preprocessing
input_data = np.array([[2.1,-1.9,5.5], [-1.5,2.4,3.5], [0.5,-7.9,5.6], [5.9,2.3,-5.8]])
data_binarized = sklearn.preprocessing.Binarizer(threshold = 0.5).transform(input_data)
print("mean = ", input_data.mean(axis = 0))
print("Std devilation = ", input_data.std(axis = 0))


import imp
import numpy as np
import sklearn.preprocessing
input_data = np.array([[2.1,-1.9,5.5], [-1.5,2.4,3.5], [0.5,-7.9,5.6], [5.9,2.3,-5.8]])
data_binarized = sklearn.preprocessing.Binarizer(threshold = 0.5).transform(input_data)
data_scaler_minmax = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print ("\nMin max scaled data:\n", data_scaled_minmax)




import imp
import numpy as np
import sklearn.preprocessing
input_data = np.array([[2.1,-1.9,5.5], [-1.5,2.4,3.5], [0.5,-7.9,5.6], [5.9,2.3,-5.8]])
data_binarized = sklearn.preprocessing.Binarizer(threshold = 0.5).transform(input_data)
# Normalize data
data_normalized_l1 = sklearn.preprocessing.normalize(input_data, norm = 'l1')
print("\nL1 normalized data:\n", data_normalized_l1)




import imp
import numpy as np
import sklearn.preprocessing
input_data = np.array([[2.1,-1.9,5.5], [-1.5,2.4,3.5], [0.5,-7.9,5.6], [5.9,2.3,-5.8]])
data_binarized = sklearn.preprocessing.Binarizer(threshold = 0.5).transform(input_data)
# Normalize data
data_normalized_l2 = sklearn.preprocessing.normalize(input_data, norm = 'l2')
print("\nL2 normalized data:\n", data_normalized_l2)

'''

import imp
import numpy as np
from sklearn import preprocessing
input_labels = ['red','black','red','green','black','yellow','white']
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)
test_labels = ['green','red','black']
encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))