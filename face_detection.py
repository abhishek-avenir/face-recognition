
import cv2
import cvlib
import dlib
import math
import os

from argparse import ArgumentParser
from glob import glob
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
	
	def __init__(self, image=None, video=None, method=None,
				 capture_image=False, capture_video=False, 
				 max_width=512, max_height=512, show_faces=False):

		self.METHODS = {
			'cascade': self.cascade,
			'hog': self.hog,
			'cnn': self.cnn,
			'cvlib': self.cvlib,
			'mtcnn': self.mtcnn}

		self.max_height = max_height
		self.max_width = max_width
		self.show_faces = show_faces
		self.faces = []
		self.confidences = []

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

	def mark_boxes(self, image):
		h, w = image.shape[:2]
		image_copy = image.copy()
		i = 0
		for box in self.faces:
			x1, y1, x2, y2 = box
			cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
			if self.confidences:
				print(self.confidences[i])
				cv2.putText(image_copy, '{0:.2f}'.format(self.confidences[i]),
							(x1-5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
							(255, 0, 0), 1, cv2.LINE_AA)
				i += 1
		cv2.imshow("Faces", image_copy)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def crop_faces(self, image):
		i = 1
		if not os.path.exists("faces/"):
			os.makedirs("faces/")
		else:
			files = glob("faces/*")
			for file in files:
				os.remove(file)
		for face in self.faces:
			x1, y1, x2, y2 = face
			x1 = int(x1 * self.ratio)
			x2 = int(x2 * self.ratio)
			y1 = int(y1 * self.ratio)
			y2 = int(y2 * self.ratio)
			cv2.imwrite(f"faces/face_{i}.jpg", image[y1:y2, x1:x2])
			i += 1

	def filter_boxes(self, face):
		x1, y1, x2, y2 = face
		if ((x1 < 0) or (x1 >= self.width) or (x2 <= x1) or (x2 >= self.width)
				or (y1 < 0) or (y1 >= self.height) or (y2 <= y1) or
				(y2 >= self.height)):
			return False
		return True

	def detect_faces(self, image=None, video=None):
		if image:
			image = cv2.imread(image)
			self.height, self.width = image.shape[:2]
			resized_image = self.resize_image(image)
			print('Processing Image...')
			s = time()
			self.METHODS[method](resized_image)
			print(f"Time taken: {time()-s} secs")
			# Filter box if the bounding box falls out of image dimensions
			self.faces = list(filter(self.filter_boxes, self.faces))
			if self.show_faces:
				self.mark_boxes(resized_image)
			self.crop_faces(image)
		elif video:
			print("Processing video...")
			frames = self.process_video(video)
			print("Writing marked video...")
			filename = os.path.join(os.path.dirname(video),
									f"marked_{os.path.basename(video)}")
			self.create_video(frames, filename)
			print(f"File saved at {filename}")

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

	def resize_image(self, image):
		h, w = image.shape[:2]
		original_height, original_width = h, w
		if h > self.max_height:
			r = math.ceil(h//self.max_height)
			w = int(w // r)
			h = int(h // r)
			image = cv2.resize(image, (w, h))
		if w > self.max_width:
			r = math.ceil(w//self.max_width)
			w = int(w // r)
			h = int(h // r)
			image = cv2.resize(image, (w, h))
		self.ratio = original_width/w
		return image

	def cascade(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = self.faceCascade.detectMultiScale(
			gray, scaleFactor=1.1, minNeighbors=5,
			flags=cv2.CASCADE_SCALE_IMAGE)
		for (x, y, w, h) in faces:
			self.faces.append([x, y, x+w, y+h])

	def hog(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		for face in self.faceDetect(gray, 1):
			(x, y, w, h) = face_utils.rect_to_bb(face)
			self.faces.append([x, y, x+w, y+h])

	def cnn(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		for rect in self.dnnFaceDetector(gray, 1):
			x1 = rect.rect.left()
			y1 = rect.rect.top()
			x2 = rect.rect.right()
			y2 = rect.rect.bottom()
			self.faces.append([x1, y1, x2, y2])

	def cvlib(self, image):
		h, w = image.shape[:2]
		faces, confidences = cvlib.detect_face(image)
		i = 0
		for face in faces:
			x1, y1, x2, y2 = face
			self.faces.append([x1, y1, x2, y2])
			self.confidences.append(confidences[i])
			i += 1

	def mtcnn(self, image):
		h, w = image.shape[:2]
		faces = self.mtcnn_detector.detect_faces(
			cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		for face in faces:
			x1, y1, width, height = face['box']
			x1 = abs(x1)
			y1 = abs(y1)
			x2, y2 = x1 + width, y1 + height
			self.faces.append([x1, y1, x2, y2])
			self.confidences.append(face['confidence'])


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('--max-height', type=int, default=512)
	parser.add_argument('--max-width', type=int, default=512)
	parser.add_argument('--show-faces', action='store_true')
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
	f = FaceDetector(method=method,
					 capture_image=args.capture_image,
					 capture_video=args.capture_video,
					 max_width=args.max_width, max_height=args.max_height,
					 show_faces=args.show_faces)
	f.detect_faces(args.image, args.video)
