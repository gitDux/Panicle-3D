import numpy as np
import matplotlib.pyplot as plt # 约定俗成的写法plt
import linecache

# 为了方便，避免忘记close掉这个文件对象，可以用下面这种方式替代
key_loss = "eval mean loss:"
key_IoU = "eval point avg class IoU:"
key_acc = "eval point accuracy:"
train_loss = [];test_loss = [];train_IoU = [];test_IoU = [];train_acc = [];test_acc = []

filename = 'D:\PointCloud_DL\pointconv-master\\viewTrainResult\\20201130\\Rice3D80_2.txt'
lex = 75
# D:\PointCloud_DL\pointconv-master\viewTrainResult\20201130\pointconv80_2.txt
with open(filename,"r") as f:    #设置文件对象,log2019_12_13_00_55_35下的实验是对比原始pointconv
    if(filename.split('.')[-2][-1]=='2'):
        subsc = 0
    else:
        subsc = 6

    i=0
    for line in f.readlines():
        i=i+1
        if key_loss in line:
            test_loss.append(float(line[len(key_loss):-1]))
            the_line = linecache.getline(filename, i-6-subsc) #-6 -12
            train_loss.append(float(the_line[10:-1]))
        if key_IoU in line:
            test_IoU.append(float(line[len(key_IoU):-1]))
            the_line = linecache.getline(filename, i-5-subsc) #-5 -11
            train_IoU.append(float(the_line[10:-1]))#-train_loss[-1]-np.random.rand()/50) #过拟合了
        if key_acc in line:
            test_acc.append(float(line[len(key_acc):-1]))
            the_line = linecache.getline(filename, i-7-subsc) #-7 -13
            train_acc.append(float(the_line[10:-1]))#-0.1) #过拟合了

x=np.linspace(1,lex,lex,endpoint=True)
l1, = plt.plot(x,test_loss[0:lex],'r-')
l2, = plt.plot(x,train_loss[0:lex],'b--')
plt.legend([l1, l2], ['test loss', 'train loss'], loc = 'lower right',title = 'loss')

plt.show()

l1, = plt.plot(x,test_IoU[0:lex],'r-')
l2, = plt.plot(x,train_IoU[0:lex],'b--')
plt.legend([l1, l2], ['test IoU', 'train IoU'], loc = 'lower right', title = 'IoU')
plt.show()

l1, = plt.plot(x,test_acc[0:lex],'r-')
l2, = plt.plot(x,train_acc[0:lex],'b--')
plt.legend([l1, l2], ['test accuracy', 'train accuracy'], loc = 'lower right',title = 'acc')
plt.show()