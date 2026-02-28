# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 12:18:45 2026

@author: Egor
"""

# ЗАДАНИЕ 2

import pandas as pd # библиотека pandas нужна для работы с данными
import matplotlib.pyplot as plt # matplotlib для построения графиков
import numpy as np # numpy для работы с векторами и матрицами

# Считываем данные 
# df = pd.read_csv('https://archive.ics.uci.edu/ml/'
#     'machine-learning-databases/iris/iris.data', header=None)

df = pd.read_csv('data.csv')

y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)

X = df.iloc[:, [0, 1, 2]].values

def neuron(w,x):
    if((w[1]*x[0]+w[2]*x[1]+w[3]*x[2]+w[0])>=0):
        predict = 1
    else: 
        predict = -1
    return predict

w = np.random.random(4)
eta = 0.01  # скорость обучения
w_iter = [] # пустой список, в него будем добавлять веса, чтобы потом построить график
for xi, target, j in zip(X, y, range(X.shape[0])):
    predict = neuron(w,xi)   
    w[1:] += (eta * (target - predict)) * xi # target - predict - это и есть ошибка
    w[0] += eta * (target - predict)
    # каждую 10ю итерацию будем сохранять набор весов в специальном списке
    if(j%10==0):
        w_iter.append(w.copy())

# посчитаем ошибки
sum_err = 0
for xi, target in zip(X, y):
    predict = neuron(w,xi) 
    sum_err += (target - predict)/2
print("Всего ошибок: ", sum_err)

xl=np.linspace(min(X[:,0]), max(X[:,0]))
yl=np.linspace(min(X[:,1]), max(X[:,1]))
x_grid, y_grid = np.meshgrid(xl, yl)
# 3d
fig3d = plt.figure()
ax3d = fig3d.add_subplot(projection='3d')
ax3d.scatter(X[y==1, 0], X[y==1, 1], X[y==1, 2], color='red', marker='o')
ax3d.scatter(X[y==-1, 0], X[y==-1, 1], X[y==-1, 2], color='blue', marker='x')

for i,w in zip(range(len(w_iter)), w_iter):
    if i > 0:
        ax3d.collections[-1].remove() # удаляем предыдущую плоскость перед отрисовкой новой
    z_grid = -(x_grid*w[1]+y_grid*w[2]+w[0])/w[3]
    ax3d.plot_surface(x_grid, y_grid, z_grid, alpha = 0.3, color = np.random.random(3))
    plt.pause(1)