#-*- coding: utf-8 -*-
from numpy import *
from scipy.ndimage import filters
from pylab import *
from PIL import Image
import os

def process_image(imagename,resultname,params="--edge-thresh 10 --peak-thresh 5"):
	# process an image and save the results in a file"""
	path = "/home/zhu/download/vlfeat-0.9.20/bin/glnx86/sift "
	if imagename[-3:] != 'pgm':
		#create a pgm file
		im = Image.open(imagename).convert('L')
		im.save('tmp.pgm')
		imagename = 'tmp.pgm'
	cmmd = str("sift "+imagename+" --output="+resultname+" "+params)
	os.system(cmmd)
	print 'processed', imagename, 'to', resultname

def read_features_from_file(filename):
	#read feature properties and return in matrix form"""
	f = loadtxt(filename)
	return f[:,:4],f[:,4:] # feature locations, descriptors

def write_features_to_file(filename,locs,desc):
	#save feature location and descriptor to file"""
	savetxt(filename,hstack((locs,desc)))

def plot_features(im,locs,circle=False):
# show image with features. input: im (image as array),locs (row, col, scale, orientation of each feature) """	
	def draw_circle(c,r):
		t = arange(0,1.01,.01)*2*pi
		x = r*cos(t) + c[0]
		y = r*sin(t) + c[1]
		plot(x,y,'b',linewidth=2)
		imshow(im)
	if circle:
#		for p in locs:
#			draw_circle(p[:2],p[2])
		[draw_circle([p[0],p[1]],p[2]) for p in locs]
	else:
		plot(locs[:,0],locs[:,1],'ob')
	axis('off')

def match(desc1,desc2):
	# for each descriptor in the first image,	select its match in the second image.input: desc1 (descriptors for the first image),desc2 (same for second image).
	desc1 = array([d/linalg.norm(d) for d in desc1])
	desc2 = array([d/linalg.norm(d) for d in desc2])								
	dist_ratio = 0.6
	desc1_size = desc1.shape															
	matchscores = zeros((desc1_size[0],1))
	desc2t = desc2.T #precompute matrix transpose
	for i in range(desc1_size[0]):
		dotprods = dot(desc1[i,:],desc2t) #vector of dot products
		dotprods = 0.9999*dotprods
		#inverse cosine and sort, return index for features in second image
		indx = argsort(arccos(dotprods))																										
		#check if nearest neighbor has angle less than dist_ratio times 2nd
		if arccos(dotprods)[indx[0]] < dist_ratio * arccos(dotprods)[indx[1]]:
			matchscores[i] = int(indx[0])																											
	return matchscores

def match_twosided(desc1,desc2):
	#two-sided symmetric version of match(). """
	matches_12 = match(desc1,desc2)
	matches_21 = match(desc2,desc1)
	ndx_12 = matches_12.nonzero()[0]
	#remove matches that are not symmetric
	for n in ndx_12:
		if matches_21[int(matches_12[n])] != n:
			matches_12[n] = 0
	return matches_12

def appendimages(im1,im2):
	#返回将两幅图像并排拼接成的一幅新图像
	#选取具有最少行数的图像，然后填充足够的空行
	rows1 = im1.shape[0]
	rows2 = im2.shape[0]
	if rows1 < rows2:
		im1 = concatenate((im1,zeros((rows2-rows1,im1.shape[1]))),axis=0)
	elif rows1 > rows2:
		im2 = concatenate((im2,zeros((rows1-rows2,im2.shape[1]))),axis=0)
	# 如果这些情况都没有，那么他们的行数相同，不需要进行填充
	return concatenate((im1,im2), axis=1)

def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
	#显示一幅带有连接匹配之间连线的图片
	#输入：im1,im2（数组图像），locs1,locs2（特征位置），matchscores（match（）的输出），
	#show_below（如果图像应该显示在匹配的下方）
	im3 = appendimages(im1,im2)
	if show_below:
		im3 = vstack((im3,im3))
	imshow(im3)
	cols1 = im1.shape[1]
	for i in range(len(matchscores)):
		if matchscores[i]>0:
			plot([locs1[i,0], locs2[matchscores[i,0],0]+cols1], [locs1[i,1], locs2[matchscores[i,0],1]], 'c')			
	axis('off')

im1f = 'left01.jpg'#'./stereo/Bowling/view1s.jpg'
im2f = 'right01.jpg'#'./stereo/Bowling/view5s.jpg'

im1 = array(Image.open(im1f))
im2 = array(Image.open(im2f))

process_image(im1f, 'image1.sift')
l1, d1 = read_features_from_file('image1.sift')
figure()
gray()
subplot(121)
plot_features(im1, l1, circle=False)

process_image(im2f, 'image2.sift')
l2, d2 = read_features_from_file('image2.sift')
subplot(122)
plot_features(im2, l2, circle=False)

matches = match_twosided(d1, d2)
#print '{} matches'.format(len(matches.nonzero()[0]))

figure()
gray()
plot_matches(im1,im2,l1,l2,matches,show_below=True)
#show()

imname = '../../imlist/empire.jpg'
im = array(Image.open(imname).convert('L'))
process_image(imname, 'image.sift')
l1,d1 = read_features_from_file('image.sift')

figure()
gray()
subplot(131)
plot_features(im, l1, circle=False)
title(u'SIFT_feature')
subplot(132)
plot_features(im,l1,circle=True)
title(u'circle_sift_feature')
show()
