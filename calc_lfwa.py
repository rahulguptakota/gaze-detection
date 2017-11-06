import sys
if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()


import os
import dlib
import glob
import matplotlib.pyplot as plt
from skimage import io
predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

i=0;
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

# for root, dirnames, filenames in os.walk(faces_folder_path):
# 	for filename in filenames:
# 		print(os.path.join(root, filename))
n=1;
break_flag = 0;
for root, dirnames, filenames in os.walk(faces_folder_path):
	for filename in filenames:
		f = os.path.join(root, filename)

		print("Processing file: {}".format(f))
		img = io.imread(f)
		i = i+1
		print i
		if(i==n): break_flag = 1;
		dets = detector(img, 1)
		print("Number of faces detected: {}".format(len(dets)))
		for k, d in enumerate(dets):
			print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
				k, d.left(), d.top(), d.right(), d.bottom()))
			# Get the landmarks/parts for the face in box d.
			shape = predictor(img, d)

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
		for i in range(37,38):
		    # print dir(shape.part(i))
		    plt.plot(shape.part(i).x, shape.part(i).y, 'o')
	        #print("Part 0: {}, Part 1: {}, Part 3: {} ...".format(shape.part(0),                                                  shape.part(1), shape.part(2)))
	        # Draw the face landmarks on the screen.
	        #win.add_overlay(shape)
	    plt.show()
	    #win.add_overlay(dets)
	    # dlib.hit_enter_to_continue()
		if(break_flag==1): break
	if(break_flag): break


print "left eye position: ", left_eye_left_x/n,",", left_eye_left_y/n, " ", left_eye_right_x/n, ",", left_eye_right_y/n, " \n", "right eye position: ", right_eye_left_x/n, ",", right_eye_left_y/n, " ", right_eye_right_x/n, ",", right_eye_right_y/n, " \n", "nose tip: ", nose_tip_x/n, ",", nose_tip_y/n, " \n", "mouth:", mouth_left_x/n, ",", mouth_left_y/n," ", mouth_right_x/n,",", mouth_right_y/n

# print("left eye position: ", left_eye_left_x/n,",", left_eye_left_y/n, " ", left_eye_right_x/n, ",", left_eye_right_y/n, " \n",
# 		"right eye position: ", right_eye_left_x/n, ",", right_eye_left_y/n, " ", right_eye_right_x/n, ",", right_eye_right_y/n, " \n", 
# 		"nose tip", nose_tip_x/n, ",", nose_tip_y/n, " \n"
# 		"mouth:", mouth_left_x/n, ",", mouth_left_y/n," ", mouth_right_x/n,",", mouth_right_y/n)
