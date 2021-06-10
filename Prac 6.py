  
import matplotlib.pyplot as plt
import neurolab as nl

#Вводим значения ввода, так как это пример контролируемого обучения, то указываем целевые значения
input = [[0, 0], [0, 1], [1, 0], [1, 1]]
target = [[0], [0], [0], [1]]

#создаем сеть с 2-мя входами и 1-им нейроном 
net = nl.net.newp([[0,1], [0,1],], 1) #Для создания однослойного персептрона предназначена функция .newp()

#тренируем сеть по правилу delta (метод обучения перцептрона по принципу градиентного спуска по поверхности ошибки)
error_progress = net.train(input, target, epochs=100, show=10, lr=0.1) #функция обучения .train()

#визуализируем вывод
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.grid()#сетка 
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

input_data = np.loadtxt("/Users/Elian/neural_simple.txt")

#разделим четыре столбца на 2 столбца данных и 2 метки
data = input_data[:, 0:2]
labels = input_data[:, 2:]

#График ввода данных
plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data') 

#определим минимальное и максимальное значения для каждого измерения
dim1_min, dim1_max = data[:,0].min(), data[:,0].max()
dim2_min, dim2_max = data[:,1].min(), data[:,1].max()

#определим количество нейронов
nn_output_layer = labels.shape[1]

#определим однослойную нейронную сеть
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
neural_net = nl.net.newp([dim1, dim2], nn_output_layer)

#Тренируем нейронную сеть с количеством эпох и скоростью обучения
error = neural_net.train(data, labels, epochs = 200, show = 20, lr = 0.01)

#визуализируем и наносим на график прогресс тренировки
plt.figure()
plt.plot(error)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.title('Training error progress')
plt.grid()
plt.show()

#используем тестовые данные в приведенном выше классификаторе
print('\nTest Results:')
data_test = [[1.5, 3.2], [3.6, 1.7], [3.6, 5.7],[1.6, 3.9]] 
for item in data_test:
    print(item, '-->', neural_net.sim([item])[0])
#сгенерируем несколько точек данных на основе уравнения: y = 2x^(2)+8
import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

#создаем точку данных на основе вышеупомянутого уравнения
min_val = -30 #минимальное значение графика по x
max_val = 30 #максимальное значение графика по x
num_points = 160 #кол-во элементов для графика
x = np.linspace(min_val, max_val, num_points) #linspace генерирует num_points точек, равномерно распределенных в интервале от min_val до max_val.
y = 2 * np.square(x) + 8 #уравнение np.square() квадрат некого числа 
y /= np.linalg.norm(y) #np.linalg.norm() используется для вычисления нормы вектора или матрицы

#изменяем этот набор данных
data = x.reshape(num_points, 1) #reshape() изменяет форму массива без изменения его данных
labels = y.reshape(num_points, 1) #reshape() изменяет форму массива без изменения его данных

#визуализируем x1)
plt.figure()
plt.scatter(data, labels)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Data-points')

"""
создаем нейронную сеть, имеющую два скрытых слоя с нейролабом
с десятью нейронами в первом скрытом слое, шесть во втором скрытом
слое и один в выходном слое
"""
neural_net = nl.net.newff([[min_val, max_val]], [10, 6, 1]) #[min_val, max_val] - диапазоны входных сигналов

#используем алгоритм обучения градиентному спуску
neural_net.trainf = nl.train.train_gd

#обучаем сеть 
error = neural_net.train(data, labels, epochs = 1000, show = 100, goal = 0.01)
"""
data, labels - обучающие множества
epochs - число циклов обучения
show - период вывода информации о состоянии процесса
goal - цель обучения, значение функционала ошибки при котором 
обучение будет завершено преждевременно
"""

#испытание сети на учебных точках данных
output = neural_net.sim(data)
y_pred = output.reshape(num_points)

#визуализация x2))
plt.figure()
plt.plot(error)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')

#отображение фактического результата в сравнении с прогнозируемым
x_dense = np.linspace(min_val, max_val, num_points * 2)
y_dense_pred = neural_net.sim(x_dense.reshape(x_dense.size, 1)).reshape(x_dense.size)

#визуализация x3)))
plt.figure()
plt.plot(x_dense, y_dense_pred, '-', x, y, '.', x, y_pred, 'p')
plt.title('Actual vs predicted')

plt.show()

'''2.  4.  0.  0. 
1.5 3.9 0.  0. 
2.2 4.1 0.  0. 
1.9 4.7 0.  0. 
5.4 2.2 0.  1. 
4.3 7.1 0.  1. 
5.8 4.9 0.  1. 
6.5 3.2 0.  1. 
3.  2.  1.  0. 
2.5 0.5 1.  0. 
3.5 2.1 1.  0. 
2.9 0.3 1.  0. 
6.5 8.3 1.  1. 
3.2 6.2 1.  1. 
4.9 7.8 1.  1. 
2.1 4.8 1.  1. 
'''