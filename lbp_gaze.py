import sys
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


import os
import dlib
import glob
import matplotlib.pyplot as plt
from skimage import io
from skimage import transform as tf
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
import numpy as np
import pygame
import pygame.camera
import time
from pygame.locals import *


pygame.init()
pygame.camera.init()

predictor_path = sys.argv[1]


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

#this is the area for customisations
frames = 50
webcam_length = 1280
webcam_width = 720
display_length = 250
display_width = 250
crop_length = 480
proc_length = 250
proc_width = 250


#-----------------------------

dst = np.array([[95 , 114],[154 , 113], [125 , 147], [106 , 161], [161 , 160]])
dst2 = np.array([[95 , 114],[154 , 113], [125 , 147]])
dstr = np.array([[95.0*(640.0/250), 114.0*(480.0/250)], [154*(640.0/250) , 113*(480.0/250)], [125*(640.0/250) , 147*(480.0/250)], [106*(640.0/250) , 161*(480.0/250)], [161*(640.0/250) , 160*(480.0/250)]])
# ,[114*(640.0/250) , 114*(480.0/250)], [137*(640.0/250) , 114*(480.0/250)]

# figure how to take input from webcam 
i=0;
displaysize = (display_length+webcam_length,webcam_width)
size = (display_length+webcam_length ,webcam_width)
# create a display surface. standard pygame stuff
display = pygame.display.set_mode(displaysize, 0)

# this is the same as what we saw before
clist = pygame.camera.list_cameras()
cam = pygame.camera.Camera(clist[0], (webcam_length, webcam_width))
cam.start()

snapshot = pygame.Surface((webcam_length, webcam_width))
snapshot2 = pygame.Surface((webcam_length, webcam_width))
# snapshot = pygame.surface.Surface(size, 0, display)


# snapshot2 = pygame.surface.Surface(size, 1, display)

# or 
# camlist = pygame.camera.list_cameras()
# if camlist:
# 	cam = pygame.caemra.Camera(camlist[0],(640,480))

# following commands can be helpful:
# cam.start()
# image = cam.get_image()
# but there is the problem of some buffer with get_image() function , so be careful
# cam.stop()
# imglew = pygame.surfarray.array3d(image)
# imglew2 = imglew.astype('unit8')
# we can use imglew2 for face detection and subsequent eye patch extraction



count = 0
# for input from webcam into img
while(1):
	starttime3 = time.time()
	snapshot = cam.get_image(snapshot)
	display.blit(snapshot, (0,0))
	# pygame.display.update()
	imglew = pygame.surfarray.array3d(snapshot)
	endtime3 = time.time()
	print("time for getting image from webcam: {}".format(endtime3 - starttime3))
	imglew = imglew[(webcam_length-crop_length)/2 : (webcam_length+crop_length)/2]
	imglew = tf.resize(imglew, (250, 250, 3))
	io.imsave("input_image.jpg", imglew)
	print imglew.shape
	# imglew2 = imglew.astype('uint8')
	input_image = tf.rotate(imglew, 270, 1)
	# io.imsave("input_image.jpg", input_image)
	test_image = input_image*255
	test_image = test_image.astype('uint8')
	


	# io.imsave("input_image.jpg", input_image)
	# io.imsave("imglew2", imglew2);
	# input_image = io.imread("custom_images/Mukesh.jpg")
	# input_image = io.imread("input_image.jpg")
	
	# print input_image
	# print test_image
	# print (input_image[1]==test_image[1])
	# print input_image - test_image
	# print input_image.shape
	starttime1 = time.time()
	dets = detector(test_image, 1)
	endtime1 = time.time()
	print("time taken to detect face: {}".format(endtime1-starttime1))
	print("Number of faces detected: {}".format(len(dets)))
	if(len(dets) == 0):
		count = count + 1
		if(count > frames):
			pygame.display.quit()
			pygame.quit()
			break
		continue
	left_eye_left_x = 0
	left_eye_left_y = 0
	left_eye_right_x = 0
	left_eye_right_y = 0
	right_eye_left_x = 0
	right_eye_left_y = 0
	right_eye_right_x = 0
	right_eye_right_y = 0
	nose_tip_x = 0
	nose_tip_y = 0
	mouth_left_x = 0
	mouth_left_y = 0
	mouth_right_x = 0
	mouth_right_y = 0
	for k, d in enumerate(dets):
		print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
			k, d.left(), d.top(), d.right(), d.bottom()))
		# Get the landmarks/parts for the face in box d.
		shape = predictor(input_image, d)

	# left eye = 36, 39    ; right eye = 42,45 ; nose tip = 33 ;   mouth left = 48;  mouth right = 54;
	
		left_eye_left_x = left_eye_left_x + shape.part(36).x
		left_eye_left_y = left_eye_left_y + shape.part(36).y
		left_eye_right_x = left_eye_right_y + shape.part(39).x
		left_eye_right_y = left_eye_right_y + shape.part(39).y
		right_eye_left_x = right_eye_left_x + shape.part(42).x
		right_eye_left_y = right_eye_left_y + shape.part(42).y
		right_eye_right_x = right_eye_right_x + shape.part(45).x
		right_eye_right_y = right_eye_right_y + shape.part(45).y
		nose_tip_x = nose_tip_x + shape.part(33).x
		nose_tip_y = nose_tip_y + shape.part(33).y
		mouth_left_x = mouth_left_x + shape.part(48).x
		mouth_left_y = mouth_left_y + shape.part(48).y
		mouth_right_x = mouth_left_y + shape.part(54).x
		mouth_right_y = mouth_right_y + shape.part(54).y

	# affine transform
	# rescale the image to 250 * 250 or take the ratio of dest with img resolution
	# print [[shape.part(48).x, shape.part(48).y], [shape.part(54).x, shape.part(54).y]]
	src = np.array([[left_eye_left_x, left_eye_left_y], [right_eye_right_x, right_eye_right_y], [nose_tip_x, nose_tip_y], [shape.part(48).x, shape.part(48).y], [shape.part(54).x, shape.part(54).y]])
	src2 = np.array([[left_eye_left_x, left_eye_left_y], [right_eye_right_x, right_eye_right_y], [nose_tip_x, nose_tip_y]])
	# , [shape.part(48).x, shape.part(48).y], [shape.part(54).x, shape.part(54).y]
	# , [shape.part(39).x, shape.part(39).y], [shape.part(42).x,shape.part(42).y ]
	starttime2 = time.time()
	# tform = tf.estimate_transform('polynomial', dst, src, order=1)
	tform = tf.estimate_transform('affine', dst, src)
	# do something with np.allclose(tform.inverse(tform(src)), src). It checks that difference between between argument matrices is negligible
	
	img2 = tf.warp(input_image, inverse_map=tform)
	endtime2 = time.time()
	print("time taken for transformtations: {}".format(endtime2 - starttime2))
	img2 = np.array(img2)
	# print img2.shape

	output_image = tf.rotate(img2, 90, 1)
	output_image = tf.resize(output_image, (proc_length, proc_width, 3))

	output_image = output_image*255
	# print output_image
	# print output_image.shape
	# io.imsave('output.jpg', output_image)
	
	# output_image = np.array(output_image)
	# output_image.shape
	# time.sleep(2)
	# pygame.surfarray.use_arraytype('numpy')
	newsur = pygame.surfarray.make_surface(output_image)
	# display.blit(snapshot2, (640, 0))

	# snapshot2 = pygame.image.load('output.jpg')
	# snapshot2 = pygame.image.load('output.jpg').convert()

	# pygame.display.update()
	display.blit(newsur, (webcam_length, 0))
	pygame.display.update()
	# viewer = ImageViewer(img2)
	# viewer.show()
	count = count + 1
	grandtimeend = time.time()
	print("time required for one iteration: {}".format(grandtimeend-starttime3))
	if(count > frames):
		pygame.display.quit()
		pygame.quit()
		break

	# io.imsave("input_classifier.jpg", img2)

	# invoke classifier on eye patch

	# place pointer based on the output