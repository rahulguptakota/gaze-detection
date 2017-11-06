import sys
import time
import random


try:
	import imfeats
except:
	pass
import imfeats
import dlib
import numpy as np
import skimage.color
from skimage import transform as tf
from skimage import feature, io
from skimage.exposure import equalize_hist
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pygame
import pygame.camera
from localbinarypattern import LocalBinaryPatterns
from pygame.locals import *
from screeninfo import get_monitors

if len(sys.argv) != 2:
	print(
		"Give the path to the trained shape predictor model as the first "
		"argument and then run again\n"
		"For example, if you are in the python_examples folder then "
		"execute this program by running:\n"
		"    python affine_trans.py shape_predictor_68_face_landmarks.dat\n"
		"You can download a trained facial shape predictor from:\n"
		"    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
	exit()



#this is the area for customisations
frames = 50
webcam_length = 1280
webcam_width = 720
display_length = 250
display_width = 250
crop_length = 480
proc_length = 250
proc_width = 250
eye_length = 70
eye_width = 35

#-----------------------------------

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
 
	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		# lbp = feature.local_binary_pattern(image, self.numPoints,
		# 	self.radius)
		image = image*255.0/(np.max(image))
		lbp = imfeats.lbp(image.astype('uint8'), np.array([7 , 3], 'int'))
		# (hist, _) = np.histogram(lbp.ravel(),
		# 	bins=np.arange(0, self.numPoints + 3),
		# 	range=(0, self.numPoints + 2))
 
		# normalize the histogram
		hist = lbp.astype("float")
		hist /= (hist.sum() + eps)
		hist = np.sqrt(hist)
		print hist.shape
 
		# return the histogram of Local Binary Patterns
		return hist
#-----------------------------

#this is area for where several modules are initalised
pygame.init()
pygame.camera.init()
predictor_path = sys.argv[1]
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
desc = LocalBinaryPatterns(24, 8)
#--------------------------------

#initialising data structures used 
data = []
labels = []
i=1
pred_i = 0
pred_arr = [1,1,1,1,1]
quad= {}
quad[1]=0
quad[2]=0
quad[3]=0
quad[4]=0
#---------------------------------

#getting info about screen resolution and camera resolution
scr = []
for m in get_monitors():
	scr = str(m)
scr_x = scr.split('(')[1].split('x')[0]
scr_y = scr.split('(')[1].split('x')[1].split('+')[0]
screen_x = int(scr_x)
screen_y = int(scr_y)
#----------------------------------

#initialising variables and surfaces for pygame display
screen = pygame.display.set_mode((screen_x, screen_y))
background = pygame.Surface(screen.get_size())
background.fill((0, 0, 0))     # fill the background white
background = background.convert()  # prepare for faster blitting
ballsurface = pygame.Surface((50, 50))     # create a rectangular surface for the ball
#pygame.draw.circle(Surface, color,, pos, radius, width=0)
# draw blue filled circle on ball surface
pygame.draw.circle(ballsurface, (255, 255, 255), (25, 25), 20)
snapshot = pygame.Surface((webcam_length, webcam_width))
ballsurface = ballsurface.convert()
#getting camera details and starting it
clist = pygame.camera.list_cameras()
cam = pygame.camera.Camera(clist[0], (webcam_length, webcam_width))
#clist species the list of path of cameras connected
#if another camera is connected clist[i], i corresponding to that camera should be used
cam.start()
#-----------------------------------
fig,ax = plt.subplots(1)

while(i<25):
	ran = random.random()
	delta = ran*(0.1)
	delta = delta - 0.05
	if(i%4+1==4):
		ballx = (0.9+ delta)*screen_x
		bally = (0.8 + delta)*screen_y
		quadrant = 4
	elif(i%4+1==3):
		ballx = (0.07 + delta)*screen_x
		bally = (0.8 + delta)*screen_y
		quadrant = 3
	elif(i%4+1==2):
		ballx = (0.05 + delta)*screen_x
		bally = (0.05 + delta)*screen_y
		quadrant = 2
	elif(i%4+1==1):
		ballx = (0.9 + delta)*screen_x
		bally = (0.07 + delta)*screen_y
		quadrant = 1

	screen.blit(background, (0, 0))
	# blit the background on the screen (overwriting all)
	screen.blit(ballsurface, (ballx, bally))
	# blit the bottomright corner of ball surface at pos (ballx, bally)
	pygame.display.update()

	#giving time to blink and handling lag that occurs often
	starttime = time.time()
	while(time.time()-starttime<1.25):
		snapshot = cam.get_image(snapshot)

	imglew = pygame.surfarray.array3d(snapshot)

	#get image , turn dlib on, detect landmarks and crop eyes, store them against correct label
	#test_image is used for detecting faces and features
	#input_image is used for cropping eyes etc
	imglew = imglew[(webcam_length-crop_length)/2 : (webcam_length+crop_length)/2]
	imglew = tf.rotate(imglew, 270, 1)
	io.imsave("img1.jpg", imglew)
	input_image = tf.resize(imglew, (720,720,3))
	test_image = tf.resize(imglew, (250, 250, 3))
	io.imsave("img2.jpg", test_image)
	test_image = test_image*255
	input_image = input_image*255
	input_image = input_image.astype('uint8')
	test_image = test_image.astype('uint8')
	# test_image = equalize_hist(test_image)
	dets = detector(test_image, 1)
	
	# print("time taken to detect face: {}".format(endtime1-starttime1))
	print("Number of faces detected: {}".format(len(dets)))

	if(len(dets) == 0):
		print("Please improve to a better lighted location or position your head in front of the webcam for face detection")
		time.sleep(1)
		continue

	for k, d in enumerate(dets):
		print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
			k, d.left(), d.top(), d.right(), d.bottom()))
		# Get the landmarks/parts for the face in box d.
		shape = predictor(test_image, d)

	ax.imshow(test_image)
	# Create a Rectangle patch
	rect = patches.Rectangle((d.left(), d.top()), d.right() - d.left(), 0 - d.top() + d.bottom() ,linewidth=1,edgecolor='r',facecolor='none')

	# Add the patch to the Axes
	ax.add_patch(rect)
	plt.show()
	fig,ax = plt.subplots(1)
	ax.imshow(test_image)
	rect = patches.Rectangle((d.left(), d.top()), d.right() - d.left(), 0 - d.top() + d.bottom() ,linewidth=1,edgecolor='r',facecolor='none')
	for i in np.array([36, 39, 42, 45, 33, 48, 54]):
		ax.add_patch(patches.Rectangle(shape.part(i).x, shape.part(i).y, 1, 1,linewidth=1, edgecolor='r',facecolor='none'))
	plt.show()
	#getting the bounding box for left and right eyes
	eyel_x1 = int((720.0/250)*(shape.part(36).x ))- ( shape.part(39).x - shape.part(36).x)/5
	eyel_x2 = int((720.0/250)*(shape.part(39).x ))+ ( shape.part(39).x - shape.part(36).x)/5
	eyel_y1 = int((720.0/250)*(shape.part(36).y ))- (eyel_x2 - eyel_x1)/4
	eyel_y2 = int((720.0/250)*(shape.part(39).y ))+ (eyel_x2 - eyel_x1)/4

	eyer_x1 = int((720.0/250)*(shape.part(42).x)) - ( shape.part(45).x - shape.part(42).x)/5
	eyer_x2 = int((720.0/250)*(shape.part(45).x)) + ( shape.part(45).x - shape.part(42).x)/5
	eyer_y1 = int((720.0/250)*(shape.part(42).y)) - (eyer_x2 - eyer_x1)/4
	eyer_y2 = int((720.0/250)*(shape.part(45).y)) + (eyer_x2 - eyer_x1)/4

	grey_image = skimage.color.rgb2gray(input_image)

	eyel = grey_image[eyel_y1:(eyel_y2+1), eyel_x1:(eyel_x2+1)]
	eyer = grey_image[eyer_y1:(eyer_y2+1), eyer_x1:(eyer_x2+1)]
	eyel = tf.resize(eyel, (eye_width, eye_length))
	eyer = tf.resize(eyer, (eye_width, eye_length))
	# eyel = equalize_hist(eyel)
	# eyer = equalize_hist(eyer)
	eye = np.hstack((eyel, eyer))
	hist1 = desc.describe(eyel)
	hist2 = desc.describe(eyer)

	io.imsave('genTrainingImages/'+str(quadrant)+'/'+str(i)+'l.jpg', eyel)
	io.imsave('genTrainingImages/'+str(quadrant)+'/'+str(i)+'r.jpg', eyer)
	# eye = equalize_hist(eye)
	io.imsave('genTrainingImages/'+str(quadrant)+'/'+str(i)+'concat.jpg', eye)
	hist = np.hstack((hist1, hist2))
	print hist
	data.append(hist)
	labels.append(quadrant)
	i = i + 1

 
# train a Linear SVM on the data
pygame.display.quit()
model = LinearSVC(C=10.0, random_state=42, multi_class="ovr")
model.fit(data, labels)

#saving svm model for future use
joblib.dump(model,"my_models/eyel.pkl")

#testing our application
# screen=pygame.display.set_mode((screen_x, screen_y))
# background = pygame.Surface(screen.get_size())
# background.fill((0,0,0))     # fill the background white
# background = background.convert()  # prepare for faster blitting
# ballsurface = pygame.Surface((50,50))     # create a rectangular surface for the ball
# #pygame.draw.circle(Surface, color,, pos, radius, width=0)
# # draw blue filled circle on ball surface
# pygame.draw.circle(ballsurface, (255,255,255), (25,25),20)
# snapshot = pygame.Surface((webcam_length, webcam_width))
# ballsurface = ballsurface.convert()

# start = time.time()
# while(time.time()-start<0):
# 	snapshot = cam.get_image(snapshot)
# 	imglew = pygame.surfarray.array3d(snapshot)

# 	#get image , turn dlib on, detect landmarks and crop eyes, store them against correct label
# 	#test_image is used for detecting faces and features
# 	#input_image is used for cropping eyes etc
# 	imglew = imglew[(webcam_length-crop_length)/2 : (webcam_length+crop_length)/2]
# 	imglew = tf.rotate(imglew, 270, 1)
# 	input_image = tf.resize(imglew, (720,720,3))
# 	test_image = tf.resize(imglew, (250, 250, 3))
# 	test_image = test_image*255
# 	input_image = input_image*255
# 	input_image = input_image.astype('uint8')
# 	test_image = test_image.astype('uint8')
# 	dets = detector(test_image, 1)
	
# 	# print("time taken to detect face: {}".format(endtime1-starttime1))
# 	print("Number of faces detected: {}".format(len(dets)))
	
# 	if(len(dets) == 0):
# 		print("Please improve to a better lighted location or position your head in front of the webcam for face detection")
# 		# time.sleep(1)
# 		continue

# 	for k, d in enumerate(dets):
# 		print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
# 			k, d.left(), d.top(), d.right(), d.bottom()))
# 		# Get the landmarks/parts for the face in box d.
# 		shape = predictor(test_image, d)

# 	#getting the bounding box for left and right eyes
# 	eyel_x1 = int((720.0/250)*(shape.part(36).x ))- ( shape.part(39).x - shape.part(36).x)/5
# 	eyel_x2 = int((720.0/250)*(shape.part(39).x ))+ ( shape.part(39).x - shape.part(36).x)/5
# 	eyel_y1 = int((720.0/250)*(shape.part(36).y ))- (eyel_x2 - eyel_x1)/4
# 	eyel_y2 = int((720.0/250)*(shape.part(39).y ))+ (eyel_x2 - eyel_x1)/4

# 	eyer_x1 = int((720.0/250)*(shape.part(42).x)) - ( shape.part(45).x - shape.part(42).x)/5
# 	eyer_x2 = int((720.0/250)*(shape.part(45).x)) + ( shape.part(45).x - shape.part(42).x)/5
# 	eyer_y1 = int((720.0/250)*(shape.part(42).y)) - (eyer_x2 - eyer_x1)/4
# 	eyer_y2 = int((720.0/250)*(shape.part(45).y)) + (eyer_x2 - eyer_x1)/4

# 	grey_image = skimage.color.rgb2gray(input_image)

# 	eyel = grey_image[eyel_y1:(eyel_y2+1), eyel_x1:(eyel_x2+1)]
# 	eyer = grey_image[eyer_y1:(eyer_y2+1), eyer_x1:(eyer_x2+1)]
# 	eyel = tf.resize(eyel, (eye_width, eye_length))
# 	eyer = tf.resize(eyer, (eye_width, eye_length))
# 	eye = np.hstack((eyel, eyer))
# 	eye = equalize_hist(eye)
# 	hist = desc.describe(eye) #histogram equilisation
# 	#using our model to predict a
# 	a = model.predict(hist)[0]
# 	print hist
# 	#predicting based on majority vote in last 5 predictions
# 	pred_arr[pred_i] = a
# 	for key in quad:
# 		quad[key] = 0
# 	for item in pred_arr:
# 		quad[item] = quad[item] + 1
# 	maxkey =0
# 	maxvalue=0
# 	for key,value in quad.items():
# 		if(maxvalue<value):
# 			maxkey = key
# 			maxvalue = value
# 	print a, maxkey, pred_arr, pred_i
# 	pred_i = (pred_i+1)%5
# 	a=maxkey
# 	if(a==4):
# 		ballx = 0.9*screen_x
# 		bally = 0.8*screen_y
# 	elif(a==3):
# 		ballx = 0.1*screen_x
# 		bally = 0.8*screen_y
# 	elif(a==2):
# 		ballx = 0.1*screen_x
# 		bally = 0.1*screen_y
# 	elif(a==1):
# 		ballx = 0.9*screen_x
# 		bally = 0.1*screen_y

# 	screen.blit(background, (0,0))     # blit the background on the screen (overwriting all)
# 	screen.blit(ballsurface, (ballx, bally))  # blit the bottomright corner of ball surface at pos (ballx, bally)
# 	pygame.display.update()

# pygame.display.quit()
pygame.quit()
