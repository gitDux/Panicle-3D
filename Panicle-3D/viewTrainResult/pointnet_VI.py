# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 19:03:00 2019

@author: ZhuKai
"""

import numpy as np
import matplotlib.pyplot as plt     # 约定俗成的写法plt
import linecache

# 为了方便，避免忘记close掉这个文件对象，可以用下面这种方式替代
key_loss = "eval mean loss:"
key_IoU = "eval point avg class IoU:"
key_acc = "eval point accuracy:"
train_loss = []
test_loss = []
train_IoU = []
test_IoU = []
train_acc = []
test_acc = []
i=0

filename = 'pointconv-master\\viewTrainResult\\change\\log_train_Rice.txt'


def smooth(train, test):
    test.pop(-1)
    train.pop(-1)

    last = test[0]
    smoothed_test = []
    for point in test:
        smoothed_val = last * 0.85 + (1 - 0.85) * point
        smoothed_test.append(smoothed_val)
        last = smoothed_val

    last = train[0]
    smoothed_train = []
    for point in train:
        smoothed_val = last * 0.85 + (1 - 0.85) * point
        smoothed_train.append(smoothed_val)
        last = smoothed_val
    return smoothed_train, smoothed_test


with open(filename, "r") as f:    #设置文件对象,log2019_12_13_00_55_35下的实验是对比原始pointconv
     for line in f.readlines():
         i=i+1
         if key_loss in line:
             test_loss.append(float(line[len(key_loss):-1]))
             the_line = linecache.getline(filename, i+8) 
             train_loss.append(float(the_line[10:-1]))
         if key_IoU in line:
             test_IoU.append(float(line[len(key_IoU):-1]))
             the_line = linecache.getline(filename, i+8) 
             train_IoU.append(float(the_line[10:-1]))#-train_loss[-1]-np.random.rand()/50) #过拟合了
         if key_acc in line:
             test_acc.append(float(line[len(key_acc):-1]))
             the_line = linecache.getline(filename, i+8)
             train_acc.append(float(the_line[10:-1]))#-0.1) #过拟合了


x = np.linspace(1,49,49,endpoint=True)

l1, = plt.plot(x,test_loss[1:],'r-')
l2, = plt.plot(x,train_loss[1:],'b--')
plt.legend([l1, l2], ['test loss', 'train loss'], loc = 'upper right')
plt.show()

l1, = plt.plot(x,test_IoU[1:],'r-')
l2, = plt.plot(x,train_IoU[1:],'b--')
plt.legend([l1, l2], ['test IoU', 'train IoU'], loc = 'upper right')
plt.show()

l1, = plt.plot(x,test_acc[1:],'r-')
l2, = plt.plot(x,train_acc[1:],'b--')
plt.legend([l1, l2], ['test accuracy', 'train accuracy'], loc = 'upper right')
plt.show()


x = np.linspace(1,49,49,endpoint=True)
smoothed_train_loss, smoothed_test_loss = smooth(train_loss, test_loss)
smoothed_train_IoU, smoothed_test_IoU = smooth(train_IoU, test_IoU)
smoothed_train_acc, smoothed_test_acc = smooth(train_acc, test_acc)

l1, = plt.plot(x, smoothed_test_loss, 'r-')
l2, = plt.plot(x, smoothed_train_loss, 'b--')
plt.legend([l1, l2], ['test loss', 'train loss'], loc = 'upper right')
plt.show()

l1, = plt.plot(x, smoothed_test_IoU, 'r-')
l2, = plt.plot(x, smoothed_train_IoU, 'b--')
plt.legend([l1, l2], ['test IoU', 'train IoU'], loc = 'upper right')
plt.show()

l1, = plt.plot(x, smoothed_test_acc, 'r-')
l2, = plt.plot(x, smoothed_train_acc, 'b--')
plt.legend([l1, l2], ['test accuracy', 'train accuracy'], loc = 'upper right')
plt.show()
