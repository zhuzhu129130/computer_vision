#-*- coding:utf-8 -*-
from PIL import Image
from numpy import *
from pylab import *
from scipy.ndimage import filters
im = array(Image.open('../house.jpg'))
print im.shape,im.dtype
im = array(Image.open('../house.jpg').convert('L'),'f')
print(im.shape,im.dtype)

im = array(Image.open('../house.jpg').convert('L'))
im2 = 255 - im
im3 = (100.0/255)*im+100
im4 = 255.0*(im/255.0)**2

print int(im.min()),int(im.max())
print int(im2.min()), int(im2.max())
print int(im3.min()), int(im3.max())
print int(im4.min()), int(im4.max())

figure()
gray()
subplot(1, 3, 1)
imshow(im2)
axis('off')
#title('convert')

subplot(1, 3, 2)
imshow(im3)
axis('off')
#title(r'$f(x)=\frac{100}{255}x+100$')

subplot(1, 3, 3)
imshow(im4)
axis('off')
#title(r'$f(x)=255(\frac{x}{255})^2$')
#show()

def imresize(im,sz):
	pil_im = Image.fromarray(uint8(im))
	return array(pil_im.resize(sz))

def histeq(im,nbr_bins = 256):
	imhist,bins = histogram(im.flatten(),nbr_bins,normed = True)
	cdf = imhist.cumsum()
	cdf = 255*cdf/cdf[-1]
	im2 = interp(im.flatten(),bins[:-1],cdf)
	return im2.reshape(im.shape),cdf
im = array(Image.open('../house.jpg').convert('L'))
im2,cdf = histeq(im)

figure()
subplot(2, 2, 1)
axis('off')
gray()
title(u'origin_image')
imshow(im)

subplot(2, 2, 2)
axis('off')
title(u'histeq_image')
imshow(im2)

subplot(2, 2, 3)
axis('off')
title(u'hist_origin')
#hist(im.flatten(), 128, cumulative=True, normed=True)
hist(im.flatten(), 128, normed=True)

subplot(2, 2, 4)
axis('off')
title(u'hist_histeq')
#hist(im2.flatten(), 128, cumulative=True, normed=True)
hist(im2.flatten(), 128, normed=True)

#show()

def compute_average(imlist):
	#计算图像列表的平均图像
	#打开第一符图像，将其存储在浮点型数组中
	averageim = array(Image.open(imlist[0]),'f')
	for imname in imlist[1:]:
		try:
			averageim += array(Image.open(imname))
		except:
			print(imname + '...skipped')
	average /= len(imlist)

	#返回uint8类型的平均图像
	return array(averageim,'uint8')

def pca(X):
	#主成分分析：
	#输入：矩阵X，其中该矩阵中存储训练数据，每一行为一条训练数据
	#返回：投影矩阵（按照维度的重要性排序），方差和均值

	#获取维数
	num_data,dim = X.shape

	#数据中心化
	mean_X = X.mean(axis = 0)
	X = X - mean_X

	if dim > num_data:
		#PCA-使用紧致技巧
		M = dot(X,X.T)#协方差矩阵
		e,EV = linalg.eigh(M)#特征值和特征向量
		tmp = dot(X.T,EV).T#这就是紧致技巧
		V = tmp[::-1]#由于最后的特征向量是我们所需要的，所以需要将其逆转
		S = sqrt(e)[::-1]#由于特征值是按照递增顺序排列的，所以需要将其逆转
		for i in range(V.shape[1]):
			V[:,i] /= S
	else:
		#PCA-使用SVD方法
		U,S,V = linalg.svd(X)
		V = V[:num_data]#仅仅返回前num_data维的数据才合理

	#返回投影矩阵，方差和均值
	return V,S,mean_X
im = array(Image.open(imlist[0]))
m,n = im.shape[0:2]
imnbr = len(imlist)

#创建矩阵，保存所有压平后的图像数据
immatrix = array([array(Image.open(im)).flatten() for im in imlist],'f')
#执行PCA操作
V,S,immean = pca.pca(immatrix)

#显示一些图像（均值图像和前7个模式）
figure()
gray()
subplot(2.4.1)
imshow(immean.reshape(m,n))
for i in range(7):
	subplot(2,4,i+2)
	imshow(V[i].reshape(m,n))
#show()


