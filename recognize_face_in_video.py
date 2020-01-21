
import cv2
import face_recognition as fr
import numpy as np
import os
import pickle

from argparse import ArgumentParser
from config import CONFIG
from utils import WIDTH, HEIGHT, get_face_encodings_from_image, load_encodings
from recognize_face import recoginize_face_in_image


def recoginize_face_in_video(
		path_name_encodings, encodings=None, threshold=0.5,
		use_cvlib=False, use_large_model=False, frame_resize=0.25):

	video_capture = cv2.VideoCapture(0)
	# Initialize some variables
	process_this_frame = True

	while True:
		# Grab a single frame of video
		ret, frame = video_capture.read()

		# Resize frame of video by `frame_resize` for faster face recognition processing
		small_frame = cv2.resize(frame, (0, 0), fx=frame_resize, fy=frame_resize)

		# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
		rgb_small_frame = small_frame[:, :, ::-1]

		# Only process every other frame of video to save time
		if process_this_frame:
			# Find all the faces and face encodings in the current frame of video
			image = recoginize_face_in_image(
				frame, path_name_encodings, encodings=encodings,
				threshold=threshold, use_cvlib=use_cvlib,
				use_large_model=use_large_model)

		process_this_frame = not process_this_frame

		# Display the resulting image
		cv2.imshow('Video', image[:, :, ::-1])

		# Hit 'q' on the keyboard to quit!
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Release handle to the webcam
	video_capture.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('-c', "--use-cvlib", action='store_true')
	parser.add_argument('-l', "--use-large-model", action='store_true')
	parser.add_argument('-t', "--threshold", default=0.5, type=float)
	parser.add_argument('-r', "--frame-resize", default=0.25, type=float)
	args = parser.parse_args()
	
	detect_model = ('use_cvlib' if args.use_cvlib else 'default')
	model_type = ('large' if args.use_large_model else 'small')

	encodings_pickle = CONFIG[detect_model][model_type]

	path_name_encodings = load_encodings(encodings_pickle)
	encodings = [x[-1] for x in path_name_encodings]
	
	recoginize_face_in_video(
		path_name_encodings, encodings, args.threshold, args.use_cvlib,
		args.use_large_model, args.frame_resize)