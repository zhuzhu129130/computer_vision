# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
from pylab import *
from scipy.ndimage import filters

im = array(Image.open('../house.jpg').convert('L'),'f')
print('中文')#
#im2 = filters.gaussian_filter(im,5)
figure()
gray()
axis('off')
subplot(1, 4, 1)
axis('off')
title(u'origin_image')
imshow(im)

for bi, blur in enumerate([2, 5, 10]):
	im2 = zeros(im.shape)
#	for i in range(3):
#		im2[:,:,i] = filters.gaussian_filter(im[:,:,i],blur)
	im2 = filters.gaussian_filter(im, blur)
	im2 = np.uint8(im2)
	imNum=str(blur)
	subplot(1, 4, 2 + bi)
	axis('off')
	title(u'st_minus'+imNum)
	imshow(im2)					
#如果是彩色图像，则分别对三个通道进行模糊
imx = zeros(im.shape)
filters.sobel(im,1,imx)
subplot(1, 4, 2)
axis('off')
title(u'(b)x_dir_diff')
imshow(imx)

imy = zeros(im.shape)
filters.sobel(im,0,imy)
subplot(1, 4, 3)
axis('off')
title(u'(c)y_dir_diff')
imshow(imy)

#mag = numpy.sqrt(imx**2 + imy**2)
mag = 255-sqrt(imx**2 + imy**2)
subplot(1, 4, 4)
title(u'(d)-MAG')
axis('off')
imshow(mag)

def imx(im, sigma):
	imgx = zeros(im.shape)
	filters.gaussian_filter(im, sigma, (0, 1), imgx)
	return imgx
def imy(im, sigma):
	imgy = zeros(im.shape)
	filters.gaussian_filter(im, sigma, (1, 0), imgy)
	return imgy
def mag(im, sigma):
	# there's also gaussian_gradient_magnitude()
	#mag = numpy.sqrt(imgx**2 + imgy**2)
	imgmag = 255 - sqrt(imgx ** 2 + imgy ** 2)
	return imgmag
sigma = [2, 5, 10]
for i in  sigma:
	subplot(3, 4, 4*(sigma.index(i))+1)
	axis('off')
	imshow(im)
	imgx=imx(im, i)
	subplot(3, 4, 4*(sigma.index(i))+2)
	axis('off')
	imshow(imgx)
	imgy=imy(im, i)
	subplot(3, 4, 4*(sigma.index(i))+3)
	axis('off')
	imshow(imgy)
	imgmag=mag(im, i)
	subplot(3, 4, 4*(sigma.index(i))+4)
	axis('off')
	imshow(imgmag)

def denoise(im,U_init,tolerance=0.1,tau=0.125,tv_weight=100):
	#使用A.Chambolle的计算步骤实现ROF去噪模型
	#输入：含有噪声的输入图像，U的初始值，TV正则项权值，步长，停业条件
	#输出：去噪和去除纹理后的图像纹理残留
	m,n = im.shape #噪声图像的大小

	#初始化
	U = U_init
	Px = im #对偶域的x分量
	Py = im #对偶域的y分量
	error = 1

	while(error>tolerance):
		Uold = U
		#原始变量的梯度
		GradUx = roll(U,-1,axis=1)-U
		GradUy = roll(U,-1,axis=0)-U
		#更新对偶变量
		PxNew = Px + (tau/tv_weight)*GradUx
		PyNew = Py + (tau/tv_weight)*GradUy
		NormNew = maximum(1,sqrt(PxNew**2+PyNew**2))

		Px = PxNew/NormNew #更新x分量(对偶)
		Py = PyNew/NormNew #更新y分量(对偶)
        #更新原始变量
		RxPx = roll(Px,1,axis=1) #对x分量进行向右x轴平移
		RyPy = roll(Py,1,axis=0) #对y分量进行向右y轴平移 
	   

		Divp = (Px-RxPx)+(Py-RyPy) #对偶域的散度
		u = im + tv_weight*Divp #更新原始变量
        #更新误差
		error = linalg.norm(U-Uold)/sqrt(n*m)

	return U,im-U #去噪后的图像和纹理残余

im = zeros((500,500))
im[100:400,100:400] = 128
im[200:300,200:300] = 255
im = im + 30*np.random.standard_normal((500,500))

U,T = denoise(im,im)
G = filters.gaussian_filter(im,10)

#保存生成结果
from scipy.misc import imsave
imsave('synth_rof.pdf',U)
imsave('synth_gaussian.pdf',G)

im = array(Image.open('../house.jpg').convert('L'))
U,T = denoise(im,im)

figure()
gray()
imshow(U)
axis('equal')
axis('off')
show()

