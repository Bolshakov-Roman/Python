import sklearn
from sklearn.datasets import load_breast_cancer #импорт набора данных

data = load_breast_cancer()#эта команда загружает набор данных

label_names = data['target_names']#с помощью следующих команд мы создаем для каждого важного набора переменную
labels = data['target']
feature_names = data['feature_names']
features = data['data']

print(label_names) #печатаем метки классов

print(labels[0])#показываем бинарность

print(feature_names[0])#две команды, приведенные ниже, создадут имена и значения функций

print(features[0])

from sklearn.model_selection import train_test_split#с помощью этих команд мы разделяем данные в наборах

train, test, train_labels, test_labels = train_test_split(features, labels, test_size = 0.40, random_state = 42)

from sklearn.naive_bayes import GaussianNB#импортируем модуль
gnb = GaussianNB()#инициализируем модель

model = gnb.fit(train, train_labels)#обучаем модель, подгоняя её к данным 

preds = gnb.predict(test)#с помощью этой команды оцениваем модель на тестовых данных
print(preds)

from sklearn.metrics import accuracy_score#с помощью этих команд определяем точность
print(accuracy_score(test_labels, preds))

#наивная байесовская модель
import sklearn
from sklearn.datasets import load_breast_cancer#импортируем набор данных
data = load_breast_cancer()# загружаем этот набор

label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
print(label_names)#эта команда печатает имена классов

print(labels[0])#показывает бинарность

print(feature_names[0])#создаем имена и их значения

print(features[0])

from sklearn.model_selection import train_test_spli#этот код импортирует функцию train_test_split из sklearn
train, test, train_labels, test_labels = train_test_split(features, labels, test_size = 0.40, random_state = 42)#с помощью этой команды разделяем данные на данные обучения и тестирования

from sklearn.naive_bayes import GaussianNB#импортируем модуль
gnb = GaussianNB()

model = gnb.fit(train, train_labels)

preds = gnb.predict(test)
print(preds)

from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels,preds))

import pydotplus
import collections
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X = [[165,19],[175,32],[136,35],[174,65],[141,28],[176,15],[131,32],
[166,6],[128,32],[179,10],[136,34],[186,2],[126,25],[176,28],[112,38],
[169,9],[171,36],[116,25],[196,25]]

Y = ['Man','Woman','Woman','Man','Woman','Man','Woman','Man','Woman',
'Man','Woman','Man','Woman','Woman','Woman','Man','Woman','Woman','Man']
data_feature_names = ['height','length of hair']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.40, random_state = 5)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

prediction = clf.predict([[133,37]])
print(prediction)

dot_data = tree.export_graphviz(clf, feature_names = data_feature_names, out_file = None, filled = True, rounded = True)
graph = pydotplus.graph_from_dot_data(dot_data)
colors = ('orange', 'yellow')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges: edges[edge].sort()

for i in range(2): dest = graph.get_node(str(edges[edge][i]))[0]
dest.set_fillcolor(colors[i])
graph.write_png('Decisiontree16.png')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
import matplotlib.pyplot as plt
import numpy as np

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 0)

forest = RandomForestClassifier(n_estimators = 50, random_state = 0)
forest.fit(X_train,y_train)

print('Accuracy on the training subset:(:.3f)',format(forest.score(X_train,y_train)))
print('Accuracy on the training subset:(:.3f)',format(forest.score(X_test,y_test)))

n_features = cancer.data.shape[1]
plt.barh(range(n_features),forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features),cancer.feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()
© 2021 GitHub, Inc.