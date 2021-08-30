# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 20:27:07 2019

@author: KaiZhu2`
"""
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

class PointCloudwithLable:
    orignPath = '../segRiceOrign/' #可以看作是静态成员变量，如果用 GenerateDataSet.orignPath和对象.orignPath不是一个值
    
    def __init__(self, name = '0'):
        self.fileNameC1 = name + '_000000.txt'# 稻穗类点云
        self.fileNameC2 = name + '_000001.txt'# 茎秆类点云
        
    #三维离散点图显示点云,s*1list
    @staticmethod
    def displayPoint(data,title):
        #解决中文显示问题
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
     
        #点数量太多不予显示
        while len(data[0]) > 20000:
        	print("点太多了！")
        	exit()
     
        #散点图参数设置
        fig=plt.figure() 
        ax=Axes3D(fig)
        ax.set_title(title) 
        ax.scatter3D(data[0], data[1],data[2], c = 'r', marker = '.') 
        ax.set_xlabel('x') 
        ax.set_ylabel('y') 
        ax.set_zlabel('z') 
        plt.show()
        
    @staticmethod
    def readXYZfile(filename, Separator, label = -1):
      data = [[], [], []]
      f = open(filename,'r') 
      line = f.readline() 
      num = 0
      while line:  #按行读入点云
         ponit = line.split(Separator)#应对包含颜色信息的情况
         data[0].append(ponit[0])  #X坐标
         data[1].append(ponit[1])  #Y坐标
         data[2].append(ponit[2])  #Z坐标
         num = num + 1
         line = f.readline()
      f.close()      
        #string型转float型 
      x = [ float(data[0] ) for data[0] in data[0] ] 
      z = [ float(data[1] ) for data[1] in data[1] ] 
      y = [ float(data[2] ) for data[2] in data[2] ]
      print("读入点的个数为:{}个。".format(num))
      if (label != -1):
          point = [x,y,z,[label]*len(x)]  #label
      else:
          point = [x,y,z]
      return point
#    def mergePointCloud(self):
      
    @staticmethod     
    def getPointCloudwithLable(name):
        pointcloud = PointCloudwithLable.readXYZfile(PointCloudwithLable.orignPath + name + '_000000.txt', ' ' , 0)
        PointCloudwithLable.displayPoint(pointcloud,"pointcloud")
        C1 = np.array(pointcloud).reshape((4,-1)).transpose(1,0);#reshape按照轴的顺序展开原来的再排序成需要的
        pointcloud = PointCloudwithLable.readXYZfile(PointCloudwithLable.orignPath + name+ '_000001.txt', ' ', 1)
        C2 = np.array(pointcloud).reshape((4,-1)).transpose(1,0);
        rice = np.append(C1,C2,axis = 0)
        np.random.shuffle(rice);#打乱点的顺序
        return rice 

if __name__ == '__main__':
    print("This is for rice point cloud segmentation, created by KaiZhu{zhuk_me@sjtu.edu.cn}")









