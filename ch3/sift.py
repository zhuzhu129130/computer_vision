from PIL import Image
import os
from numpy import *
from pylab import *
import homography
import warp

def process_image(imagename,resultname,params="--edge-thresh 10 --peak-thresh 5"):
    """ process an image and save the results in a file"""
    path = "/home/zhu/download/vlfeat-0.9.20/bin/glnx86/sift """
    if imagename[-3:] != 'pgm':
	    #create a pgm file
		 im = Image.open(imagename).convert('L')
		 im.save('tmp.pgm')
		 imagename = 'tmp.pgm'
    cmmd = str(path+imagename+" --output="+resultname+
				" "+params)
    os.system(cmmd)
    print 'processed', imagename, 'to', resultname


def read_features_from_file(filename):
	""" read feature properties and return in matrix form"""
	f = loadtxt(filename)
	return f[:,:4],f[:,4:] # feature locations, descriptors


def write_features_to_file(filename,locs,desc):
	""" save feature location and descriptor to file"""
	savetxt(filename,hstack((locs,desc)))
	

def plot_features(im,locs,circle=False):
	""" show image with features. input: im (image as array), 
		locs (row, col, scale, orientation of each feature) """

	def draw_circle(c,r):
		t = arange(0,1.01,.01)*2*pi
		x = r*cos(t) + c[0]
		y = r*sin(t) + c[1]
		plot(x,y,'b',linewidth=2)

	imshow(im)
	if circle:
		[draw_circle([p[0],p[1]],p[2]) for p in locs]
	else:
		plot(locs[:,0],locs[:,1],'ob')
	axis('off')


def match(desc1,desc2):
	""" for each descriptor in the first image, 
		select its match in the second image.
		input: desc1 (descriptors for the first image), 
		desc2 (same for second image). """
	
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


def appendimages(im1,im2):
	""" return a new image that appends the two images side-by-side."""
	
	#select the image with the fewest rows and fill in enough empty rows
	rows1 = im1.shape[0]    
	rows2 = im2.shape[0]
	
	if rows1 < rows2:
		im1 = concatenate((im1,zeros((rows2-rows1,im1.shape[1]))), axis=0)
	elif rows1 > rows2:
		im2 = concatenate((im2,zeros((rows1-rows2,im2.shape[1]))), axis=0)
	#if none of these cases they are equal, no filling needed.
	
	return concatenate((im1,im2), axis=1)


def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
	""" show a figure with lines joining the accepted matches
		input: im1,im2 (images as arrays), locs1,locs2 (location of features), 
		matchscores (as output from 'match'), show_below (if images should be shown below). """
	
	im3 = appendimages(im1,im2)
	if show_below:
		im3 = vstack((im3,im3))
	
	# show image
	imshow(im3)
	
	# draw lines for matches
	cols1 = im1.shape[1]
	for i in range(len(matchscores)):
		if matchscores[i] > 0:
			plot([locs1[i,0], locs2[matchscores[i,0],0]+cols1], [locs1[i,1], locs2[matchscores[i,0],1]], 'c')
	axis('off')


def match_twosided(desc1,desc2):
	""" two-sided symmetric version of match(). """
	
	matches_12 = match(desc1,desc2)
	matches_21 = match(desc2,desc1)
	
	ndx_12 = matches_12.nonzero()[0]
	
	#remove matches that are not symmetric
	for n in ndx_12:
		if matches_21[int(matches_12[n])] != n:
			matches_12[n] = 0
	
	return matches_12


# set paths to data folder
featname = ['../data/Univ'+str(i+1)+'.sift' for i in range(5)] 
imname = ['../data/Univ'+str(i+1)+'.jpg' for i in range(5)]
# extract features and match
l = {}
d = {}
for i in range(5):
	process_image(imname[i],featname[i])
	l[i],d[i] = read_features_from_file(featname[i])

matches = {}
for i in range(4):
	matches[i] = match(d[i+1],d[i])

# visualize the matches (Figure 3-11 in the book)
for i in range(4):
	im1 = array(Image.open(imname[i]))
	im2 = array(Image.open(imname[i+1]))
	figure()
	plot_matches(im2,im1,l[i+1],l[i],matches[i],show_below=True)

# function to convert the matches to hom. points
def convert_points(j):
	ndx = matches[j].nonzero()[0]
	fp = homography.make_homog(l[j+1][ndx,:2].T) 
	ndx2 = [int(matches[j][i]) for i in ndx]
	tp = homography.make_homog(l[j][ndx2,:2].T) 
	# switch x and y - TODO this should move elsewhere
	fp = vstack([fp[1],fp[0],fp[2]])
	tp = vstack([tp[1],tp[0],tp[2]])
	return fp,tp

# estimate the homographies
model = homography.RansacModel() 

fp,tp = convert_points(1)
H_12 = homography.H_from_ransac(fp,tp,model)[0] #im 1 to 2 

fp,tp = convert_points(0)
H_01 = homography.H_from_ransac(fp,tp,model)[0] #im 0 to 1 

tp,fp = convert_points(2) #NB: reverse order
H_32 = homography.H_from_ransac(fp,tp,model)[0] #im 3 to 2 

tp,fp = convert_points(3) #NB: reverse order
H_43 = homography.H_from_ransac(fp,tp,model)[0] #im 4 to 3    


# warp the images
delta = 2000 # for padding and translation

im1 = array(Image.open(imname[1]), "uint8")
im2 = array(Image.open(imname[2]), "uint8")
im_12 = warp.panorama(H_12,im1,im2,delta,delta)

im1 = array(Image.open(imname[0]), "f")
im_02 = warp.panorama(dot(H_12,H_01),im1,im_12,delta,delta)

im1 = array(Image.open(imname[3]), "f")
im_32 = warp.panorama(H_32,im1,im_02,delta,delta)

im1 = array(Image.open(imname[4]), "f")
im_42 = warp.panorama(dot(H_32,H_43),im1,im_32,delta,2*delta)


figure()
imshow(array(im_42, "uint8"))
axis('off')
show()
	
	
	
