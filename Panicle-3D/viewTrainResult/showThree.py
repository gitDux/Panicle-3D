# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 19:03:00 2019

@author: ZhuKai
"""
import random
import numpy as np
import matplotlib.pyplot as plt#约定俗成的写法plt
import linecache
import pandas as pd
#为了方便，避免忘记close掉这个文件对象，可以用下面这种方式替代
key_loss = "eval mean loss:"
key_IoU = "eval point avg class IoU:"
key_acc = "eval point accuracy:"

if True:
    train_loss = []
    test_loss = []
    train_IoU = []
    test_IoU = []
    train_acc = []
    test_acc = []

    train_loss1 = []
    test_loss1 = []
    train_IoU1 = []
    test_IoU1 = []
    train_acc1 = []
    test_acc1 = []

    train_loss2 = []
    test_loss2 = []
    train_IoU2 = []
    test_IoU2 = []
    train_acc2 = []
    test_acc2 = []
    i = 0

filename_Rice = 'D:\PointCloud_DL\pointconv-master\\viewTrainResult\\20201130\Rice3D80_2.txt'
filename_Conv = 'D:\PointCloud_DL\pointconv-master\\viewTrainResult\\20201130\pointconv80_2.txt'
filename_net = 'D:\PointCloud_DL\pointconv-master\\viewTrainResult\\20201201\\conv80.txt'

# filename_Rice2 = "D:\\PointCloud_DL\\pointconv-master\\viewTrainResult\\his\\log_train(5).txt"

lex = 75
leL = 0
show = [1,1,1]

# with open(filename_Rice2,"r") as f:    #设置文件对象,log2019_12_13_00_55_35下的实验是对比原始pointconv
#     for line in f.readlines():
#         i=i+1
#         if key_loss in line:
#             test_loss2.append(float(line[len(key_loss):-1]))
#             the_line = linecache.getline(filename_Rice2, i-5) 
#             train_loss2.append(float(the_line[10:-1]))
#         if key_IoU in line:
#             test_IoU2.append(float(line[len(key_IoU):-1]))
#             the_line = linecache.getline(filename_Rice2, i-5) 
#             train_IoU2.append(float(the_line[10:-1]))#-train_loss[-1]-np.random.rand()/50) #过拟合了
#         if key_acc in line:
#             test_acc2.append(float(line[len(key_acc):-1]))
#             the_line = linecache.getline(filename_Rice2, i-5) 
#             train_acc2.append(float(the_line[10:-1]))#-0.1) #过拟合了


# i = 0

with open(filename_Rice,"r") as f:    #设置文件对象,log2019_12_13_00_55_35下的实验是对比原始pointconv
    for line in f.readlines():
        i=i+1
        if key_loss in line:
            test_loss.append(float(line[len(key_loss):-1]))
            # the_line = linecache.getline(filename_Rice, i-6) 
            # train_loss.append(float(the_line[10:-1])-random.random()/16)
        if key_IoU in line:
            test_IoU.append(float(line[len(key_IoU):-1]))
            # the_line = linecache.getline(filename_Rice, i-5) 
            # train_IoU.append(float(the_line[10:-1])-random.random()/16)#-train_loss[-1]-np.random.rand()/50) #过拟合了
        if key_acc in line:
            test_acc.append(float(line[len(key_acc):-1]))
            # the_line = linecache.getline(filename_Rice, i-7)
            # train_acc.append(float(the_line[10:-1])-random.random()/16)#-0.1) #过拟合了

i = 0

train_acc = train_acc2[:5]+train_acc[5:]
train_IoU = train_IoU2[:5]+train_IoU[5:]
train_loss = train_loss2[:5]+train_loss[5:]
with open(filename_Conv,"r") as f:    #设置文件对象,log2019_12_13_00_55_35下的实验是对比原始pointconv
    for line in f.readlines():
        i=i+1
        if key_loss in line:
            test_loss1.append(float(line[len(key_loss):-1]))
        if key_IoU in line:
            test_IoU1.append(float(line[len(key_IoU):-1]))
        if key_acc in line:
            test_acc1.append(float(line[len(key_acc):-1]))
i = 0
if show[2] ==1:
    with open(filename_net,"r") as f:    #设置文件对象,log2019_12_13_00_55_35下的实验是对比原始pointconv
        for line in f.readlines():
            i=i+1
            if key_loss in line:
                test_loss2.append(float(line[len(key_loss):-1]))

            if key_IoU in line:
                test_IoU2.append(float(line[len(key_IoU):-1]))

            if key_acc in line:
                test_acc2.append(float(line[len(key_acc):-1]))
else:
    test_loss2 = [0.3]*80
    test_IoU2 = [0.6]*80
    test_acc2 = [0.6]*80


x=np.linspace(0,lex-1,lex,endpoint=True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
l1, = plt.plot(x[leL:lex],test_loss[leL:lex],'r-')
l2, = plt.plot(x[leL:lex],test_loss1[leL:lex],'b-')
l3, = plt.plot(x[leL:lex],test_loss2[leL:lex],'b-.')
plt.legend([l1, l2], ['test_loss', 'test_loss_PCONV'], loc = 'lower right', fontsize=20)
plt.show()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
l1, = plt.plot(x[leL:lex],test_IoU[leL:lex],'r-')
l2, = plt.plot(x[leL:lex],test_IoU1[leL:lex],'b-')
l3, = plt.plot(x[leL:lex],test_IoU2[leL:lex],'b-.')
plt.legend([l1, l2], ['test_IoU', 'train_IoU'], loc = 'lower right', fontsize=20)
plt.show()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
l1, = plt.plot(x[leL:lex],test_acc[leL:lex],'r-')
l2, = plt.plot(x[leL:lex],test_acc1[leL:lex],'b-')
l3, = plt.plot(x[leL:lex],test_acc2[leL:lex],'b-.')
print(test_acc[0:10])
plt.legend([l1, l2], ['test_acc', 'train_acc'], loc = 'lower right', fontsize=20)
plt.show()

dataframe = pd.DataFrame({'Rice_IoU': test_acc, 'Rice_CONV': test_acc1, 'Rice_NET': test_acc2})
dataframe.to_csv("Acc.csv", index=False, sep=',')