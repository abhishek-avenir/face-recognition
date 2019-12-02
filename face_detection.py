
import cv2
import cvlib
import dlib
import math
import os

from argparse import ArgumentParser
from imutils import face_utils
from mtcnn import MTCNN
from time import time

HAAR_CASC_PATH = os.path.join(
	os.path.dirname(cv2.__dict__['cv2'].__dict__['__file__']), "data",
	"haarcascade_frontalface_default.xml")
CNN_FACE_DETECTION_MODEL = \
	os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "dlib",
				"cnn_face_detection_model_v1",
				"mmod_human_face_detector.dat")


class FaceDetector(object):
	
	def __init__(self, image=None, video=None, capture_image=False,
				 capture_video=False, method=None):

		self.METHODS = {
			'cascade': self.cascade,
			'hog': self.hog,
			'cnn': self.cnn,
			'cvlib': self.cvlib,
			'mtcnn': self.mtcnn}
		if method not in self.METHODS:
			raise Exception(f"Method has to be one of {self.METHODS}")
			
		if method == 'cnn':
			self.dnnFaceDetector = dlib.cnn_face_detection_model_v1(
				CNN_FACE_DETECTION_MODEL)
		elif method == 'hog':
			self.faceDetect = dlib.get_frontal_face_detector()
		elif method == 'cascade':
			self.faceCascade = cv2.CascadeClassifier(HAAR_CASC_PATH)
		elif method == "mtcnn":
			self.mtcnn_detector = MTCNN()

		if image:
			image = cv2.imread(image)
			image = self.resize_image(image)
			print('Processing Image...')
			s = time()
			marked = self.METHODS[method](image)
			print(f"Time taken: {time()-s} secs")
			self.show_image(marked)
		elif video:
			print("Processing video...")
			frames = self.process_video(video)
			print("Writing marked video...")
			filename = os.path.join(os.path.dirname(video),
									f"marked_{os.path.basename(video)}")
			self.create_video(frames, filename)
			print(f"File saved at {filename}")
		# elif capture_image:
		# 	video_capture = cv2.VideoCapture(0)

	def process_video(self, video):
		frames = []
		cap = cv2.VideoCapture(video)
		while cap.isOpened():
			ret, frame = cap.read()
			if ret:
				marked = self.METHODS[method](frame)
				frames.append(marked)
			else:
				break
		cap.release()
		return frames

	def create_video(self, op_frames, filename):
		h, w = op_frames[-1].shape[:2]
		fps = 10
		out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
		for frame in op_frames:
			out.write(frame)
		out.release()

	@staticmethod
	def resize_image(image, max_width=512, max_height=1024):
		h, w = image.shape[:2]
		if h > 512:
			r = math.ceil(h//512)
			w = int(w // r)
			h = int(h // r)
			image = cv2.resize(image, (w, h))
		if w > 1024:
			r = math.ceil(w//1024)
			w = int(w // r)
			h = int(h // r)
			image = cv2.resize(image, (w, h))
		return image

	@staticmethod
	def show_image(image, window="Face", wait_for=0):
		cv2.imshow(window, image)
		cv2.waitKey(wait_for)
		cv2.destroyWindow(window)

	def cascade(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = self.faceCascade.detectMultiScale(
			gray, scaleFactor=1.1, minNeighbors=5,
			flags=cv2.CASCADE_SCALE_IMAGE)
		for (x, y, w, h) in faces:
			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
		return image

	def hog(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		for face in self.faceDetect(gray, 1):
			(x, y, w, h) = face_utils.rect_to_bb(face)
			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
		return image

	def cnn(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		for rect in self.dnnFaceDetector(gray, 1):
			x1 = rect.rect.left()
			y1 = rect.rect.top()
			x2 = rect.rect.right()
			y2 = rect.rect.bottom()
			cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
		return image

	def cvlib(self, image):
		faces, confidences = cvlib.detect_face(image)
		i = 0
		for face in faces:
			x1, y1, x2, y2 = face
			cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), -1)
			print(f"Confidence: {confidences[i]}")
			i += 1
		return image

	def mtcnn(self, image):
		faces = self.mtcnn_detector.detect_faces(
			cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		for face in faces:
			x1, y1, w, h = face['box']
			x2, y2 = x1 + w, y1 + h
			cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), -1)
			print(f"Confidence: {face['confidence']}")
		return image


if __name__ == "__main__":
	parser = ArgumentParser()
	group1 = parser.add_mutually_exclusive_group(required=True)
	group1.add_argument('-i', "--image", help='Path to the image file')
	group1.add_argument('-v', "--video", help='Path to the video file')
	group1.add_argument('-ci', "--capture-image", help='Capture image',
						action='store_true')
	group1.add_argument('-cv', "--capture-video", help='Capture video',
						action='store_true')
	group2 = parser.add_mutually_exclusive_group(required=True)
	group2.add_argument("--cascade", action='store_true')
	group2.add_argument("--hog", action='store_true')
	group2.add_argument("--cnn", action='store_true')
	group2.add_argument("--cvlib", action='store_true')
	group2.add_argument("--mtcnn", action='store_true')
	args = parser.parse_args()

	if args.cascade:
		method = 'cascade'
	elif args.hog:
		method = 'hog'
	elif args.cnn:
		method = 'cnn'
	elif args.cvlib:
		method = 'cvlib'
	elif args.mtcnn:
		method = 'mtcnn'
	f = FaceDetector(args.image, args.video, args.capture_image,
					 args.capture_video, method)
