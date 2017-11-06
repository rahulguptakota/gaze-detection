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
args = parser.parse_args()

print args