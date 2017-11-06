import sys
import os
import time
import random
import ConfigParser
import argparse

import imfeats
import dlib
import numpy as np
import skimage.color
from skimage import transform as tf
from skimage import feature, io
from skimage.exposure import equalize_hist
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

import pygame
import pygame.camera
from pygame.locals import *
from screeninfo import get_monitors

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("shape_detector_model",
                    help="Give the path of shape detector model for dlib. It is available at \
							 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
parser.add_argument("-u", "--usepretrained",
                    help="if this option is used with a path, it will use the model at given path,\
					else it will undergo training")
parser.add_argument("-t", "--train",
                    help="if this option is used with a path, it will store the model at given path,\
					else it will store it at the present working directory" )
parser.add_argument("-d", "--duration",
					help="(useful only when used with usepretrained, duration for which prediction is to be done)")
args = parser.parse_args()

print args



#this is the area for customisations
#do changes in the config file "config.ini" for customisations
Config = ConfigParser.ConfigParser()
Config.read("config.ini")

def ConfigSectionMap(section):
	dict1 = {}
	options = Config.options(section)
	for option in options:
		try:
			dict1[option] = Config.get(section, option)
			if dict1[option] == -1:
				DebugPrint("skip: %s" % option)
		except:
			print("exception on %s!" % option)
			dict1[option] = None
	return dict1

frames = int(ConfigSectionMap("options")['frames'])
webcam_length = int(ConfigSectionMap("options")['webcam_length'])
webcam_width = int(ConfigSectionMap("options")['webcam_width'])
display_length = int(ConfigSectionMap("options")['display_length'])
display_width = int(ConfigSectionMap("options")['display_width'])
crop_length = int(ConfigSectionMap("options")['crop_length'])
proc_length = int(ConfigSectionMap("options")['proc_length'])
proc_width = int(ConfigSectionMap("options")['proc_width'])
eye_length = int(ConfigSectionMap("options")['eye_length'])
eye_width = int(ConfigSectionMap("options")['eye_width'])
blink_time = float(ConfigSectionMap("options")['blink_time']) #blink_time>= 1.1, else it won't work due to webcam lag
window_length = float(ConfigSectionMap("options")['window_length'])
radius = int(ConfigSectionMap("options")['radius'])
lbp_n = int(ConfigSectionMap("options")['lbp_n'])
lbp_r = int(ConfigSectionMap("options")['lbp_r'])
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
predictor_path = args.shape_detector_model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
desc = LocalBinaryPatterns(lbp_n, lbp_r)
data = []
labels = []
#--------------------------------

#getting info about screen resolution and camera resolution
def get_scr_info():
	scr = []
	for m in get_monitors():
		scr = str(m)
	scr_x = scr.split('(')[1].split('x')[0]
	scr_y = scr.split('(')[1].split('x')[1].split('+')[0]
	screen_x = int(scr_x)
	screen_y = int(scr_y)
	return screen_x, screen_y
#----------------------------------
screen_x, screen_y = get_scr_info()

#initialising variables and surfaces for pygame display
screen = pygame.display.set_mode((screen_x, screen_y))
background = pygame.Surface(screen.get_size())
background.fill((0, 0, 0))     # fill the background white
background = background.convert()  # prepare for faster blitting
ballsurface = pygame.Surface((40, 40))     # create a rectangular surface for the ball
#pygame.draw.circle(Surface, color,, pos, radius, width=0)
# draw blue filled circle on ball surface
pygame.draw.circle(ballsurface, (255, 255, 255), (20, 20), 20)
snapshot = pygame.Surface((webcam_length, webcam_width))
ballsurface = ballsurface.convert()
shell1 = pygame.Surface((2*radius, 2*radius))
shell2 = pygame.Surface((2*radius, 2*radius))
shell3 = pygame.Surface((2*radius, 2*radius))
shell4 = pygame.Surface((2*radius, 2*radius))
#getting camera details and starting it
clist = pygame.camera.list_cameras()
cam = pygame.camera.Camera(clist[0], (webcam_length, webcam_width))
#clist species the list of path of cameras connected
#if another camera is connected clist[i], i corresponding to that camera should be used
cam.start()
#-----------------------------------
def get_snapshot(blink_time):
	starttime = time.time()
	while(time.time()-starttime<blink_time):
		img_snapshot = cam.get_image(snapshot)
	return img_snapshot

def get_ballpos(window_length, quadrant):
	ran = random.random()
	delta = ran*(window_length)
	if(quadrant==4):
		ballx = (0.95 - delta)*screen_x
		bally = (0.85 - delta)*screen_y
	elif(quadrant==3):
		ballx = (0.12 - delta)*screen_x
		bally = (0.85 - delta)*screen_y
	elif(quadrant==2):
		ballx = (0.1 - delta)*screen_x
		bally = (0.1 - delta)*screen_y
	elif(quadrant==1):
		ballx = (0.9 - delta)*screen_x
		bally = (0.12 - delta)*screen_y
	return ballx, bally

def get_eye_cord(img):
	test_image = tf.resize(img, (250, 250, 3))
	test_image = test_image*255
	test_image = test_image.astype('uint8')
	# test_image = equalize_hist(test_image)
	dets = detector(test_image, 1)
	
	# print("time taken to detect face: {}".format(endtime1-starttime1))
	print("Number of faces detected: {}".format(len(dets)))

	if(len(dets) == 0):
		print("Please move to a better lighted location or position your head in front of the webcam for face detection")
		time.sleep(1)
		return		

	for k, d in enumerate(dets):
		print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
			k, d.left(), d.top(), d.right(), d.bottom()))
		# Get the landmarks/parts for the face in box d.
		shape = predictor(test_image, d)

	#getting the bounding box for left and right eyes
	eyel_x1 = int((720.0/250)*(shape.part(36).x ))- ( shape.part(39).x - shape.part(36).x)/5
	eyel_x2 = int((720.0/250)*(shape.part(39).x ))+ ( shape.part(39).x - shape.part(36).x)/5
	eyel_y1 = int((720.0/250)*(shape.part(36).y ))- (eyel_x2 - eyel_x1)/4
	eyel_y2 = int((720.0/250)*(shape.part(39).y ))+ (eyel_x2 - eyel_x1)/4

	eyer_x1 = int((720.0/250)*(shape.part(42).x)) - ( shape.part(45).x - shape.part(42).x)/5
	eyer_x2 = int((720.0/250)*(shape.part(45).x)) + ( shape.part(45).x - shape.part(42).x)/5
	eyer_y1 = int((720.0/250)*(shape.part(42).y)) - (eyer_x2 - eyer_x1)/4
	eyer_y2 = int((720.0/250)*(shape.part(45).y)) + (eyer_x2 - eyer_x1)/4
	return eyel_x1, eyel_x2, eyel_y1, eyel_y2, eyer_x1, eyer_x2, eyer_y1, eyer_y2

def display_error():
	font = pygame.font.Font(None, int(36.0*(screen_x/1280.0)))
	text0 = font.render("Your face is not detected:", 1, (200, 200, 200))
	text1 = font.render("Please consider moving to a better lighted location or ", 1, (200, 200, 200))
	text2 = font.render("Position your head in front of the webcam for face detection", 1, (200, 200, 200))
	screen.blit(text0, (screen_x*0.25, screen_y*0.45))
	screen.blit(text1, (screen_x*0.25, screen_y*0.50))
	screen.blit(text2, (screen_x*0.25, screen_y*0.55))
	pygame.display.flip()
	time.sleep(1)

def get_feature_vector(img, quadrant = None, i = None):
	#process image before cropping eye
	try:
		eyel_x1, eyel_x2, eyel_y1, eyel_y2, eyer_x1, eyer_x2, eyer_y1, eyer_y2 = get_eye_cord(img)
	except TypeError:
		return

	input_image = tf.resize(img, (720,720,3))
	input_image = input_image*255
	input_image = input_image.astype('uint8')
	grey_image = skimage.color.rgb2gray(input_image)

	#cropping eye images
	eyel = grey_image[eyel_y1:(eyel_y2+1), eyel_x1:(eyel_x2+1)]
	eyer = grey_image[eyer_y1:(eyer_y2+1), eyer_x1:(eyer_x2+1)]
	#resize them to standard size
	eyel = tf.resize(eyel, (eye_width, eye_length))
	eyer = tf.resize(eyer, (eye_width, eye_length))
	eye = np.hstack((eyel, eyer))

	hist1 = desc.describe(eyel)
	hist2 = desc.describe(eyer)
	hist = np.hstack((hist1, hist2))
	if quadrant is not None:
		if not os.path.exists('genTrainingImages/'+str(quadrant)):
			os.makedirs('genTrainingImages/'+str(quadrant))
		io.imsave('genTrainingImages/'+str(quadrant)+'/'+str(i)+'l.jpg', eyel)
		io.imsave('genTrainingImages/'+str(quadrant)+'/'+str(i)+'r.jpg', eyer)
		io.imsave('genTrainingImages/'+str(quadrant)+'/'+str(i)+'concat.jpg', eye)
	return hist

def training():
	i=0
	while(i<8):
		quadrant = i%4 + 1
		ballx, bally = get_ballpos(window_length, quadrant)
		screen.blit(background, (0, 0))
		# blit the background on the screen (overwriting all)
		screen.blit(ballsurface, (ballx, bally))
		# blit the bottomright corner of ball surface at pos (ballx, bally)
		pygame.display.update()
		#giving time to blink and handling lag that occurs often		
		snapshot = get_snapshot(blink_time)
		img = pygame.surfarray.array3d(snapshot)

		#get image , turn dlib on, detect landmarks and crop eyes, store them against correct label
		#test_image is used for detecting faces and features
		#input_image is used for cropping eyes etc
		img = img[(webcam_length-crop_length)/2 : (webcam_length+crop_length)/2]
		img = tf.rotate(img, 270, 1)

		
		feature_vector = get_feature_vector(img, quadrant, i)#quadrant and i are given for storing training photos
		if(feature_vector is None):
			display_error()
			continue
		data.append(feature_vector)
		labels.append(quadrant)
		i = i + 1

	# train a Linear SVM on the data
	pygame.display.quit()
	model = LinearSVC(C=10.0, random_state=42, multi_class="ovr")
	model.fit(data, labels)

	#saving svm model for future use
	if(args.train is None):
		joblib.dump(model, "lbp_model.pkl")
	else:
		joblib.dump(model, args.train)



def draw_fixed_circles(radius):
	#pygame.draw.circle(Surface, color,, pos, radius, width=0)
	# draw blue filled circle on ball surface
	pygame.draw.circle(ballsurface, (255, 255, 255), (radius, radius), radius)
	pygame.draw.circle(shell1, (255, 255, 255), (radius, radius), radius, 2)
	pygame.draw.circle(shell2, (255, 255, 255), (radius, radius), radius, 2)
	pygame.draw.circle(shell3, (255, 255, 255), (radius, radius), radius, 2)
	pygame.draw.circle(shell4, (255, 255, 255), (radius, radius), radius, 2)
	shell1.convert()
	shell2.convert()
	shell3.convert()
	shell4.convert()

def predict():
	draw_fixed_circles(20)
	if(args.usepretrained is None):
		model = joblib.load("lbp_model.pkl")
	else:
		model = joblib.load(args.usepretrained)
	starttime = time.time()
	while(time.time()-starttime<duration):
		snapshot = get_snapshot(blink_time)
		img = pygame.surfarray.array3d(snapshot)

		#get image , turn dlib on, detect landmarks and crop eyes, store them against correct label
		#test_image is used for detecting faces and features
		#input_image is used for cropping eyes etc
		img = img[(webcam_length-crop_length)/2 : (webcam_length+crop_length)/2]
		img = tf.rotate(img, 270, 1)

		feature_vector = get_feature_vector(img)
		if(feature_vector is None):
			display_error()
			continue
		
		prediction = model.predict(feature_vector)[0]
		if(prediction==4):
			ballx = 0.95*screen_x
			bally = 0.85*screen_y
		elif(prediction==3):
			ballx = 0.02*screen_x
			bally = 0.85*screen_y
		elif(prediction==2):
			ballx = 0.02*screen_x
			bally = 0.03*screen_y
		elif(prediction==1):
			ballx = 0.95*screen_x
			bally = 0.03*screen_y

		screen.blit(background, (0,0))     # blit the background on the screen (overwriting all)
		screen.blit(shell1, (0.95*screen_x,0.85*screen_y))
		screen.blit(shell2, (0.02*screen_x,0.85*screen_y))
		screen.blit(shell3, (0.02*screen_x,0.03*screen_y))
		screen.blit(shell4, (0.95*screen_x,0.03*screen_y))
		screen.blit(ballsurface, (ballx, bally))  # blit the bottomright corner of ball surface at pos (ballx, bally)
		pygame.display.update()
		
if( args.duration is None):	
	duration = 25
else:
	duration = int(args.duration)

if( args.usepretrained is None):	
	training()
else:
	predict()

pygame.quit()