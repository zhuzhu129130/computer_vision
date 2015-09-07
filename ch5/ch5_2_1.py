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

#在前两个视图中点的索引
ndx = (corr[:,0]>=0)&(corr[:,1]>=0)

#获得坐标，并将其用齐次坐标表示
x1 = points2D[0][:,corr[ndx,0]]
x1 = vstack((x1,ones(x1.shape[1])))
x2 = points2D[1][:,corr[ndx,1]]
x2 = vstack((x2,ones(x2.shape[1])))

Xtrue = points3D[:,ndx]
Xtrue = vstack((Xtrue,ones(Xtrue.shape[1])))

#检查前三个点
Xset = sfm.triangulate(x1,x2,P[0].P,P[1].P)
print Xset[:,:3]
print Xtrue[:,:3]

#绘制图像
fig = figure()
ax = axes3d.Axes3D(fig)
ax.plot(Xset[0],Xset[1],Xset[2],'ko')
ax.plot(Xtrue[0],Xtrue[1],Xtrue[2],'r.')
axis('equal')

show()
