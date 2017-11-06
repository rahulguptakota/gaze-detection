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

dst = np.array([[95 , 114],[154 , 113], [125 , 147]])
dstr = np.array([[95.0*(640.0/250), 114.0*(480.0/250)],[154*(640.0/250) , 113*(480.0/250)], [125*(640.0/250) , 147*(480.0/250)]])


# figure how to take input from webcam 
i=0;
displaysize = (640*2,480)
size = (640, 480)
# create a display surface. standard pygame stuff
display = pygame.display.set_mode(displaysize, 0)

# this is the same as what we saw before
clist = pygame.camera.list_cameras()
cam = pygame.camera.Camera(clist[0], size)
cam.start()

snapshot = pygame.Surface(size)
snapshot2 = pygame.Surface(size)
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
	snapshot = cam.get_image(snapshot)
	display.blit(snapshot, (0,0))
	pygame.display.update()
	imglew = pygame.surfarray.array3d(snapshot)
	print imglew.shape
	imglew2 = imglew.astype('uint8')
	input_image = tf.rotate(imglew2, 270, 1)

	io.imsave("input_image.jpg", input_image)
	input_image = input_image.astype('uint8')
	# io.imsave("input_image.jpg", input_image)
	# io.imsave("imglew2", imglew2);
	# input_image = io.imread("custom_images/Mukesh.jpg")
	input_image = io.imread("input_image.jpg")
	print input_image.shape
	dets = detector(input_image, 1)
	print("Number of faces detected: {}".format(len(dets)))
	
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
	src = np.array([[left_eye_left_x, left_eye_left_y], [right_eye_right_x, right_eye_right_y], [nose_tip_x, nose_tip_y]])
	tform = tf.estimate_transform('affine', src, dstr)
	# do something with np.allclose(tform.inverse(tform(src)), src). It checks that difference between between argument matrices is negligible
	img2 = tf.warp(input_image, inverse_map=tform.inverse)
	img2 = np.array(img2)
	print img2.shape

	
	output_image = tf.resize(img2, (480,680, 3))
	print output_image.shape
	io.imsave('output.jpg', output_image)
	
	# output_image = np.array(output_image)
	# output_image.shape
	# time.sleep(2)
	pygame.surfarray.use_arraytype('numpy')
	newsur = pygame.surfarray.make_surface(output_image)
	# display.blit(snapshot2, (640, 0))

	snapshot2 = pygame.image.load('output.jpg')
	snapshot2 = pygame.image.load('output.jpg').convert()

	# pygame.display.update()
	display.blit(snapshot2, (640, 0))
	pygame.display.update()
	# viewer = ImageViewer(img2)
	# viewer.show()
	count = count + 1
	if(count >5):
		pygame.display.quit()
		pygame.quit()
		break

	# io.imsave("input_classifier.jpg", img2)

	# invoke classifier on eye patch

	# place pointer based on the output
