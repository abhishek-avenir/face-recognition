
# Create face_encodings for faces in `known_images`
> python prepare_reference_encodings.py -i "known_images/" [-c] [-l] [-o encodings/dir/my_encodings.pkl]

# To run the face recoginition

## On image
> python recognize_face.py -t 0.5 -i "test_img.jpg" [--use-cvlib] [--use-large-model]

## On video
> python recognize_face_in_video.py -t 0.5 -i "test_img.jpg" [--use-cvlib] [--use-large-model]
