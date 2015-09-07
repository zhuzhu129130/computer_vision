#-*- coding: utf-8 -*-
from pylab import *
from PIL import Image
from mpl_toolkits.mplot3d import axes3d
import camera

#载入一些图像
im1 = array(Image.open('../data/image_ox/images_4/001.jpg'))
im2 = array(Image.open('../data/image_ox/images_4/002.jpg'))

#载入每个视图的二维点列表
points2D = [loadtxt('../data/image_ox/images_4/2D/00'+str(i+1)+'.corners').T for i in range(3)]

#载入三维点
points3D = loadtxt('../data/image_ox/images_4/3D/p3d').T

#载入对应
corr = genfromtxt('../data/image_ox/images_4/2D/nview-corners',dtype='int',missing_values='*')

#载入照相机矩阵到Camera对象列表中
P = [camera.Camera(loadtxt('../data/image_ox/images_4/2D/00'+str(i+1)+'.P')) for i in range(3)]

#将三维点转换成齐次坐标表示，并投影
X = vstack((points3D,ones(points3D.shape[1])))
x = P[0].project(X)

#在视图1中绘制点
figure()
imshow(im1)
plot(points2D[0][0],points2D[0][1],'*')
axis('off')

figure()
imshow(im1)
plot(x[0],x[1],'r.')
axis('off')

fig = figure()
ax = axes3d.Axes3D(fig)
#测试3D图像，生成三维样本点
#X,Y,Z = axes3d.get_test_data(0.25)

#在三维中绘制点
#ax.plot(X.flatten(),Y.flatten(),Z.flatten(),'o')

ax.plot(points3D[0],points3D[1],points3D[2],'k.')
show()
