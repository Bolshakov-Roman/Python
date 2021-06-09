import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans

from sklearn.datasets.samples_generator import make_blobs

X, y_true = make_blobs(n_samples = 500, centers = 4, cluster_std = 0.40, random_state = 0)

plt.scatter(X[:, 0], X[:, 1], s = 50)
plt.show()

kmeans = KMeans(n_clusters = 4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1],c = y_kmeans, s = 50, cmap = 'viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = 'black', s = 200, alpha = 0.5);
plt.show()

import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

from sklearn.datasets.samples_generator import make_blobs

centers = [[2,2],[4,5],[3,10]]
X, _ = make_blobs(n_samples = 500, centers = centers, cluster_std = 1)
plt.scatter(X[:,0],X[:,1])
plt.show()

ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Estimated clusters:", n_clusters_)

colors = 10*['r.','g.','b','c.','k.','y.','m.']
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10 )
    plt.scatter(cluster_centers[:,0], cluster_centers[:,1], marker = "x", color = 'k', s = 150, linewidths = 5, zorder = 10)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

A = np.array([[3.1, 2.3], [2.3, 4.2], [3.9, 3.5], [3.7, 6.4], [4.8, 1.9],[8.3, 3.1], [5.2, 7.5], [4.8, 4.7], [3.5, 5.1], [4.4, 2.9],])

k = 3
test_data = [3.3, 2.9]

plt.figure()
plt.title('Input data')
plt.scatter(A[:,0], A[:,1], marker = 'o', s = 100, color = 'black')

knn_model = NearestNeighbors(n_neighbors = k, algorithm = 'auto').fit(X)
distances, indices = knn_model.kneighbors([test_data])

print('\nK Nearest Neighbors:')
for rank, index in enumerate(indices[0][:k], start = 1):
  print(str(rank) + " is", A[k])

plt.figure()
plt.title('Nearest neighbors')
plt.scatter(A[:9, 0], X[:9, 1], marker = 'o', s = 100, color = 'k')
plt.scatter(A[:k, 0], A[:k, 1], marker = 'o', s = 250, color = 'k', facecolors = 'none')
plt.scatter(test_data[0], test_data[1], marker = 'x', s = 100, color = 'k')
plt.show()

# Commented out IPython magic to ensure Python compatibility.
from sklearn.datasets import *
import pandas as pd
# %matplotlib inline
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

def Image_display(i):
  plt.imshow(digit['images'][i], cmap = 'Greys_r')
plt.show()
digit = load_digits()
digit_d = pd.DataFrame(digit['data'][0:1600])
Image_display(0)

train_x = digit['data'][:1600]
train_y = digit['target'][:1600]
KNN = KNeighborsClassifier(20)
KNN.fit(train_x, train_y)

KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski', metric_params = None, n_jobs = 1, n_neighbors = 20, p = 2, weights = 'uniform')

test = np.array(digit['data'][1725])
test1 = test.reshape(1, -1)
Image_display(1725)

KNN.predict(test1)

digit['target_names']