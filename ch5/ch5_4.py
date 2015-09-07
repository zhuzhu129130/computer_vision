#-*- coding: utf-8 -*-
from pylab import *
from PIL import Image
import stereo

im_l = array(Image.open('../data/stereo-pairs/teddy/imL.png').convert('L'),'f')
im_r = array(Image.open('../data/stereo-pairs/teddy/imR.png').convert('L'),'f')

#开始偏移并设置步长
steps = 12
start = 4

#ncc的宽度
wid = 9

res = stereo.plane_sweep_ncc(im_l,im_r,start,steps,wid)

import scipy.misc
scipy.misc.imsave('depth.png',res)

