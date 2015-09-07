#-*- coding: utf-8 -*-
from pylab import *
from PIL import Image
from mpl_toolkits.mplot3d import axes3d
import camera
import sfm

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
#X = vstack((points3D,ones(points3D.shape[1])))
#x = P[0].project(X)

corr = corr[:,0]
ndx3D = where(corr>=0)[0]#丢失的数值为-1
ndx2D = corr[ndx3D]

#选取可见点，并用齐次坐标表示
x = points2D[0][:,ndx2D] #视图1
x = vstack((x,ones(x.shape[1])))
X = points3D[:,ndx3D]
X = vstack((X,ones(X.shape[1])))

#估计P
Pest = camera.Camera(sfm.compute_P(x,X))

#比较！
print Pest.P / Pest.P[2,3]
print P[0].P / P[0].P[2,3]

xest = Pest.project(X)

#绘制图像
figure()
imshow(im1)
plot(x[0],x[1],'bo')
plot(xest[0],xest[1],'r.')
axis('off')

show()
