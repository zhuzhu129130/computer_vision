#-*- coding: utf-8 -*-
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame,pygame.image
from pygame.locals import *
import pickle

def set_projection_from_camera(K):
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()

	fx = K[0,0]
	fy = K[1,1]
	fovy = 2*arctan(0.5*height/fy)*180/pi
	aspect = (width*fy)/(height*fx)

	#定义近的远的剪裁平面
	near = 0.1
	far = 100.0

	#设定透视
	gluPerspective(fovy,aspect,near,far)
	glViewport(0,0,width,height)

def set_modelview_from_camera(Rt):
	"""从照相机姿态中获得模拟视图矩阵"""

	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()

	#围绕x轴将茶壶旋转90度，使z轴向上
	Rx = array([[1,0,0],[0,0,-1],[0,1,0]])

	#获得旋转的最佳逼近
	R =Rt[:,:3]
	U,S,V = linalg.svd(R)
	R = dot(U,V)
	R[0,:] = -R[0,:]

	#获得平移量
	t = Rt[:,3]

	#获得平移量
	t = Rt[:,3]

	#获得4x4的模拟视图矩阵
	M = eye(4)
	M[:3,:3] = dot(R,Rx)
	M[:3,3] = t

	#转置并压平以获取列序数值
	M = M.T
	m = flatten()

	#将模拟视图矩阵替换为新的矩阵
	glLoadMatrixf(m)

def draw_background(imname):

	bg_image = pygame.image.load(imname).convert()
	bg_data = pygame.image.tostring(bg_image,"RGBX",1)

	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

	glEnable(GL_TEXTURE_2D)
	glBindTexture(GL_TEXTURE_2D,glGenTextures(1))
	glTexImage2D(GL_TEXTURE_2D,0,GL_TEXTURE,width,height,0,GL_RGBA,GL_UNSIGNED_BYTE,bg_data)
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_fiLTER,GL_NEAREST)
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_fiLTER,GL_NEAREST)

	glBegin(GL_QUADS)
	glTexCoord2f(0.0,0.0);glVertex3f(-1.0,-1.0,-1.0)
	glTexCoord2f(1.0,0.0);glVertex3f(-1.0,-1.0,-1.0)
	glTexCoord2f(1.0,1.0);glVertex3f(-1.0,-1.0,-1.0)
	glTexCoord2f(0.0,1.0);glVertex3f(-1.0,-1.0,-1.0)
	glEnd()

	glDeleteTextures(1)

def draw_teapot(size):
	glEnable(GL_LIGHTING)
	glEnable(GL_LIGHTo)
	glEnable(GL_DEPTH_TEST)
	glClear(GL_DEPTH_BUFFER_BIT)

	glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0])
	glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.5,0.0,0.0,0.0])
	glMaterialfv(GL_FRONT,GL_SPECULAR,[0.7,0.6,0.6,0.0])
	glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0)
	glutSolidTeapot(size)


width,height = 1000,747

def setup():
	pygame.init()
	#pygame.display.set_mode((width,height),OPENGL | DOUBLEBUF)
	pygame.display.set_caption('OpenGL AR demo')

with open('ar_camera.pkl','r') as f:
	K = pickle.load(f)
	Rt = pickle.load(f)

	setup()
#	draw_background('../data/book_perspective.bmp')
	set_projection_from_camera(K)
	set_modelview_from_camera(Rt)
	draw_teapot(0.02)

	while True:
		event = pygame.event.poll()
		if event.type in (QUIT,KEYDOWN):
			break
	pygame.display.flip()

