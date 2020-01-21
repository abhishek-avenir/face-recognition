import cv2
import dlib
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to image file')
ap.add_argument('-w', '--weights',required = True, help = 'path to weights file')
args = ap.parse_args()

image  = cv2.imread(args.image)
if image is None:
	print("could not read input image")
	exit()

# hog_face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1(args.weights)

# # apply HOG face detection

# start = time.time()
# faces_hog = hog_face_detector(image,1)
# end = time.time()
# print("Exection Time (in seconds) :")
# print("HOG : ",format(end - start, '.2f'))
# for face in faces_hog:
# 	x = face.left()
# 	y = face.top()
# 	w = face.right() - x
# 	h = face.bottom()- y
# 	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

# apply CNN Face detection
start = time.time()
faces_cnn = cnn_face_detector(image,1)
end = time.time()
print ("CNN : ",format(end - start,'.2f'))

for face in faces_cnn:
	x = face.rect.left()
	y = face.rect.top()
	w = face.rect.right()-x
	h = face.rect.bottom()- y
	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

img_height,img_width = image.shape[:2]
# cv2.putText(image, "HOG", (img_width-50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                 (0,255,0), 2)
cv2.putText(image, "CNN", (img_width-50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
(0,0,255), 2)
cv2.imshow("face detection with dlib", image)
cv2.waitKey()

cv2.imwrite("cnn_face_detection.png", image)
cv2.destroyAllWindows()