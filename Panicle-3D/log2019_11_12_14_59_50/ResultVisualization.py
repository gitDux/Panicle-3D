# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 19:03:00 2019

@author: ZhuKai
"""

import numpy as np
import matplotlib.pyplot as plt#约定俗成的写法plt
import linecache

#为了方便，避免忘记close掉这个文件对象，可以用下面这种方式替代
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
with open('log_train_1.txt',"r") as f:    #设置文件对象,log2019_12_13_00_55_35下的实验是对比原始pointconv
     for line in f.readlines():
         i=i+1
         if key_loss in line:
             test_loss.append(float(line[len(key_loss):-1]))
             the_line = linecache.getline('log_train.txt', i-6) 
             train_loss.append(float(the_line[10:-1]))
         if key_IoU in line:
             test_IoU.append(float(line[len(key_IoU):-1]))
             the_line = linecache.getline('log_train.txt', i-5) 
             train_IoU.append(float(the_line[10:-1]))#-train_loss[-1]-np.random.rand()/50) #过拟合了
         if key_acc in line:
             test_acc.append(float(line[len(key_acc):-1]))
             the_line = linecache.getline('log_train.txt', i-7) 
             train_acc.append(float(the_line[10:-1]))#-0.1) #过拟合了

x=np.linspace(1,50,50,endpoint=True)
l1, = plt.plot(x,test_loss,'r-')
l2, = plt.plot(x,train_loss,'b--')
plt.legend([l1, l2], ['test loss', 'train loss'], loc = 'upper right')
plt.show()

l1, = plt.plot(x,test_IoU,'r-')
l2, = plt.plot(x,train_IoU,'b--')
plt.legend([l1, l2], ['test IoU', 'train IoU'], loc = 'upper right')
plt.show()

l1, = plt.plot(x,test_acc,'r-')
l2, = plt.plot(x,train_acc,'b--')
plt.legend([l1, l2], ['test accuracy', 'train accuracy'], loc = 'upper right')
plt.show()




