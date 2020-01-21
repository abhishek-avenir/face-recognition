
# Create face_encodings for faces in `known_images`

> python recognize_face.py -f "known_images/" -p
> python recognize_face.py -f "known_images/" -p --use-large-model

> python recognize_face.py -f "known_images/" -p --use-cvlib
> python recognize_face.py -f "known_images/" -p --use-cvlib --use-large-model


# To run the face recoginition

## On image
> python recognize_face.py -t 0.5 -i "test_img.jpg" [--use-cvlib] [--use-large-model]

## On video
> python recognize_face_in_video.py -t 0.5 -i "test_img.jpg" [--use-cvlib] [--use-large-model]