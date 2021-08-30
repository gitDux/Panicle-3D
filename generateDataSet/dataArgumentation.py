# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:20:45 2019

@author: KaiZhu
"""
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from dataLoadwithLabel import PointCloudwithLable
from sklearn.utils import shuffle

def rotate_point_cloud(data):
    """
    :param data: Nx3 array
    :return: rotated_data: Nx3 array
    """
    angles = np.random.uniform() * 2 * np.pi
    cosval = np.cos(angles)
    sinval = np.sin(angles)
    R = np.array([[cosval, 0, sinval],
                  [0, 1, 0],
                  [-sinval, 0, cosval]])
    rotated_data = np.dot(data, R)
    return rotated_data


def random_rotate_point_cloud(data, angle_sigma=0.06, angle_clip=0.18):
    """
    :param data: Nx3 array
    :return: rotated_data: Nx3 array
    """
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    rotated_data = np.dot(data, R)

    return rotated_data


def jitter_point_cloud(data, sigma=[0, 0, 0.01], clip=0.05):#此处x,y数据较小
    """
    :param data: Nx3 array
    :return: jittered_data: Nx3 array
    """
    N, C = data.shape
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_data += data

    return jittered_data


def random_scale_point_cloud(data, scale_low=0.8, scale_high=1.25):
    """
    :param data:  Nx3 array
    :return: scaled_data:  Nx3 array
    """
    scale = np.random.uniform(scale_low, scale_high)
    scaled_data = data * scale

    return scaled_data


def random_dropout_point_cloud(data, p=0.9):
    """
    :param data:  Nx3 array
    :return: dropout_data:  Nx3 array
    """
    N, C = data.shape
    dropout_ratio = np.random.random() * p
    drop_idx = np.where(np.random.random(N) <= dropout_ratio)[0]
    dropout_data = np.zeros_like(data)
    if len(drop_idx) > 0:
        dropout_data[drop_idx, :] = data[0, :]

    return dropout_data

def downsample_certain_num(pointcloud, num=1024):
    src_size = len(pointcloud)
    output = []
    for i in range(num):
        idx = int(np.random.random()*src_size)
        output.append(pointcloud[idx])
    return np.array(output)
    

def shift_point_cloud(data, shift_range=0.1):
    """
    :param data:  Nx3 array
    :return: shift_data:  Nx3 array
    """
    N, C = data.shape
    shifts = np.random.uniform(-shift_range, shift_range, 3)#-shift_range, shift_range范围随机数
    shift_data = data + shifts
    return shift_data

def pyplot_draw_point_cloud_with_label(points, output_filename=None):
    """ points is a Nx4 numpy array """
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #ax.set_zlim3d(-4, 4)                    # viewrange for z-axis should be [-4,4] 
    points0 =points[points[:,3] == 0]
    points1 =points[points[:,3] == 1]
    ax = Axes3D(fig)
    ax.scatter(points0[:, 0], points0[:, 1], points0[:, 2], c = 'r' , marker = '.') 
    ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c = 'b' , marker = '.') 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
#    plt.gca().set_aspect('equal', adjustable='box')
    plt.show() 

def pointCloudArgumentation(pointCloudwithLabel, pointNum, k = 100): 
    points_lable = downsample_certain_num(pointCloudwithLabel, pointNum)#随机下采样
    points = points_lable[:,0:3]
    labels = points_lable[:,3:4]
    points = (points - points.mean(axis = 0))/(points.max() - points.min())
    dataSet_X = [points]
    dataSet_Y = [labels]
    for i in range(k-1):
        points_lable = downsample_certain_num(pointCloudwithLabel, pointNum)#随机下采样
        
        points = points_lable[:,0:3]
        labels = points_lable[:,3:4]
        #归一化
        points = (points - points.mean(axis = 0))/(points.max() - points.min())
        #增强               
        points = jitter_point_cloud(points)        
        points = random_scale_point_cloud(points)
        points = random_rotate_point_cloud(points)
        points = shift_point_cloud(points)        
        #随机镜像
        if(np.random.randn() > 0):
            points = points*[-1,-1,1]
        points_lable = np.c_[points, labels]
        np.random.shuffle(points_lable)#打乱点的顺序
        dataSet_X = np.append(dataSet_X,[points_lable[:,0:3]],axis = 0)
        dataSet_Y = np.append(dataSet_Y,[points_lable[:,3:4]],axis = 0)
    return dataSet_X, dataSet_Y

def get_trainSetwithLabel(file_name, pointNum=1024, increase=100):
    dataSet_X = np.arange(pointNum*3).reshape(1,pointNum,3)
    dataSet_Y = np.arange(pointNum).reshape(1,pointNum,1)
    for d in file_name:        
       rice = PointCloudwithLable.getPointCloudwithLable(d)  
       #同时归一化
       temp_X, temp_Y = pointCloudArgumentation(rice, pointNum, increase)  
       dataSet_X = np.append(dataSet_X,temp_X,axis = 0)
       dataSet_Y = np.append(dataSet_Y,temp_Y,axis = 0)
       #打乱样本顺序
    dataSet_X = dataSet_X[1:,:,:]#丢掉第一个占位数据
    dataSet_Y = dataSet_Y[1:,:,:]
    dataSet_X, dataSet_Y = shuffle(dataSet_X, dataSet_Y, random_state=0)
    return dataSet_X, dataSet_Y
       

if __name__ == '__main__':  
    file_name = ['7']
    dataSet_X, dataSet_Y = get_trainSetwithLabel(file_name,3000,1)     
    pyplot_draw_point_cloud_with_label(np.c_[dataSet_X[0],dataSet_Y[0]])
    for i in range(len(dataSet_X)):
        points = np.array(dataSet_X[i])
        labels = np.array(dataSet_Y[i])
        p_l = np.concatenate((points,labels),axis=1)
        np.savetxt('rice_label_test_%d.txt'%i,p_l)
    
    
    
    
    

