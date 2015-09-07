#-*- coding: utf-8 -*-
import matplotlib.delaunay as md
import numpy as np
from scipy import ndimage
from pylab import *
from PIL import Image

def normalize(points):
	#在齐次做标意义下，对点集进行归一化，使最后一行为1
	for row in points:
		row /=points[-1]
	return points

def make_homog(points):
	#将点集（dim X n的数组）转化为齐次做标表示
	return vstack((points,ones((1,points.shape[1]))))

def H_from_points(fp,tp):
	#Find homography H, such that fp is mapped to tp using the linear DLT method. Points are conditioned automatically. """
	if fp.shape != tp.shape:
		raise RuntimeError('number of points do not match')	        
	# condition points (important for numerical reasons)
	# --from points--
	m = mean(fp[:2], axis=1)
	maxstd = max(std(fp[:2], axis=1)) + 1e-9
	C1 = diag([1/maxstd, 1/maxstd, 1]) 
	C1[0][2] = -m[0]/maxstd
	C1[1][2] = -m[1]/maxstd
	fp = dot(C1,fp)
	# --to points--
	m = mean(tp[:2], axis=1)
	maxstd = max(std(tp[:2], axis=1)) + 1e-9
	C2 = diag([1/maxstd, 1/maxstd, 1])
	C2[0][2] = -m[0]/maxstd
	C2[1][2] = -m[1]/maxstd
	tp = dot(C2,tp)
	# create matrix for linear method, 2 rows for each correspondence pair
	nbr_correspondences = fp.shape[1]
	A = zeros((2*nbr_correspondences,9))
	for i in range(nbr_correspondences):
		A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0,tp[0][i]*fp[0][i],tp[0][i]*fp[1][i],tp[0][i]]
		A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1,tp[1][i]*fp[0][i],tp[1][i]*fp[1][i],tp[1][i]]	
	U,S,V = linalg.svd(A)
	H = V[8].reshape((3,3)) 
	# decondition
	H = dot(linalg.inv(C2),dot(H,C1))
	# normalize and return
	return H / H[2,2]

def Haffine_from_points(fp,tp):
	#Find H, affine transformation, such that tp is affine transf of fp. """
	if fp.shape != tp.shape:
		raise RuntimeError('number of points do not match')
	# condition points
	# --from points--
	m = mean(fp[:2], axis=1)
	maxstd = max(std(fp[:2], axis=1)) + 1e-9
	C1 = diag([1/maxstd, 1/maxstd, 1]) 
	C1[0][2] = -m[0]/maxstd
	C1[1][2] = -m[1]/maxstd
	fp_cond = dot(C1,fp)
	# --to points--
	m = mean(tp[:2], axis=1)
	C2 = C1.copy() #must use same scaling for both point sets
	C2[0][2] = -m[0]/maxstd
	C2[1][2] = -m[1]/maxstd
	tp_cond = dot(C2,tp)						    
	# conditioned points have mean zero, so translation is zero
	A = concatenate((fp_cond[:2],tp_cond[:2]), axis=0)
	U,S,V = linalg.svd(A.T)
	# create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
	tmp = V[:2].T
	B = tmp[:2]
	C = tmp[2:4]			    
	tmp2 = concatenate((dot(C,linalg.pinv(B)),zeros((2,1))), axis=1) 
	H = vstack((tmp2,[0,0,1]))							    
	
	# decondition
	H = dot(linalg.inv(C2),dot(H,C1))
	return H / H[2,2]

def image_in_image(im1,im2,tp):
	#使用仿射变换将im1放置在im2上，使im1图像的角和tp尽可能的靠近，tp是齐此表示的，并且是按照从左上角逆时针计算的
	#扭曲的点
	m,n = im1.shape[:2]
	fp = array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])

	#计算仿射变换，并且将其应用于图像im1
	H = Haffine_from_points(tp,fp)
	im1_t = ndimage.affine_transform(im1,H[:2,:2],(H[0,2],H[1,2]),im2.shape[:2])
	alpha = (im1_t>0)

	return (1-alpha)*im2 + alpha*im1_t

def alpha_for_triangle(points,m,n):
	# Creates alpha map of size (m,n) for a triangle with corners defined by points (given in normalized homogeneous coordinates).
	alpha = zeros((m,n))
	for i in range(min(points[0]),max(points[0])):
		for j in range(min(points[1]),max(points[1])):
			x = linalg.solve(points,[i,j,1])
			if min(x) > 0: #all coefficients positive
				alpha[i,j] = 1
	return alpha

def triangulate_points(x,y):
	#Delaunay triangulation of 2D points. """
	centers,edges,tri,neighbors = md.delaunay(x,y)
	return tri

def pw_affine(fromim,toim,fp,tp,tri):
	#Warp triangular patches from an image.fromim = image to warp toim = destination image fp = from points in hom. coordinates tp = to points in hom.  coordinates tri = triangulation. """
	im = toim.copy()
	# check if image is grayscale or color
	is_color = len(fromim.shape) == 3
	# create image to warp to (needed if iterate colors)
	im_t = zeros(im.shape, 'uint8')
	for t in tri:
		# compute affine transformation
		H = Haffine_from_points(tp[:,t],fp[:,t])						        
		if is_color:
			for col in range(fromim.shape[2]):
				im_t[:,:,col] = ndimage.affine_transform(fromim[:,:,col],H[:2,:2],(H[0,2],H[1,2]),im.shape[:2])
		else:
			im_t = ndimage.affine_transform(fromim,H[:2,:2],(H[0,2],H[1,2]),im.shape[:2])																
		# alpha for triangle
		alpha = alpha_for_triangle(tp[:,t],im.shape[0],im.shape[1])							
		# add triangle to image
		im[alpha>0] = im_t[alpha>0]
	return im

def plot_mesh(x,y,tri):
	#Plot triangles. """ 
	for t in tri:
		t_ext = [t[0], t[1], t[2], t[0]] # add first point to end
		plot(x[t_ext],y[t_ext],'r')

# open image to warp
fromim = array(Image.open('../data/sunset_tree.jpg')) 
x, y = meshgrid(range(5), range(6))

x = (fromim.shape[1]/4) * x.flatten()
y = (fromim.shape[0]/5) * y.flatten()

# triangulate
tri = triangulate_points(x, y)

# open image and destination points
im = array(Image.open('../data/turningtorso1.jpg'))

figure()
subplot(1, 3, 1)
axis('off')
imshow(im)

tp = loadtxt('../data/turningtorso1_points.txt', 'int')  # destination points

# convert points to hom. coordinates (make sure they are of type int)
fp = array(vstack((y, x, ones((1, len(x))))), 'int')
tp = array(vstack((tp[:, 1], tp[:, 0], ones((1, len(tp))))), 'int')

# warp triangles
im = pw_affine(fromim, im, fp, tp, tri)

# plot
subplot(1, 3, 2)
axis('off')
imshow(im)
subplot(1, 3, 3)
axis('off')
imshow(im)
plot_mesh(tp[1], tp[0], tri)

show()
