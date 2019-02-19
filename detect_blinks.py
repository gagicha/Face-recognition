
# coding: utf-8

# In[14]:

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time

import dlib
import cv2
import glob
import os
import face_recognition

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
 
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
	return ear

def get_face_embeddings_from_image(image, convert_to_rgb=False):
    """
    Take a raw image and run both the face detection and face embedding model on it
    """
    # Convert from BGR to RGB if needed
    if convert_to_rgb:
        image = image[:, :, ::-1]

    # run the face detection model to find face locations
    face_locations = face_recognition.face_locations(image)
    # face_locations = face_recognition.face_locations(face_image, number_of_times_to_upsample=2)

    # run the embedding model to get face embeddings for the supplied locations
    face_encoding = face_recognition.face_encodings(image, face_locations) # , num_jitters=10

    return face_locations, face_encoding

def setup_database():
    """
    Load reference images and create a database of their face encodings
    """
    database = {}
    IMAGES_PATH= 'training_images'
    for filename in glob.glob(os.path.join(IMAGES_PATH, '*.jpeg')):
        # load image
        image_rgb = face_recognition.load_image_file(filename)
        # use the name in the filename as the identity key
        identity = os.path.splitext(os.path.basename(filename))[0]
        # get the face encoding and link it to the identity
        locations, encodings = get_face_embeddings_from_image(image_rgb)
        database[identity] = encodings[0]

    return database

def paint_detected_face_on_image(frame, face, name=None):
    """
    Paint a rectangle around the face and write the name
    """
    # unpack the coordinates from the location tuple
    top, right, bottom, left = face

    if name is None:
        name = 'Unknown'
        color = (0, 0, 255)  # red for unrecognized face
    else:
        color = (0, 128, 0)  # dark green for recognized face

    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
#     cv2.rectangle(frame, pt1, pt2, color, 2)
    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom -35), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

def run_face_recognition_video(database):
    """
    Start the face recognition via the webcam
    """
    # Open a handler for the camera
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 1

    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    video_capture = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor_model = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_model)
    known_face_encodings = list(database.values())
    known_face_names = list(database.keys())
    
    while video_capture.isOpened():
        # Grab a single frame of video (and check if it went ok)
        ok, frame = video_capture.read(0)
        if not ok:
            logging.error("Could not read frame from camera. Stopping video capture.")
            break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        locations_list=[]
        blinks_list=[]
        face_locations = detector(frame, 0)
        for i,location in enumerate(face_locations):
            shape = predictor(frame, location)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
          #  cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
          # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear < EYE_AR_THRESH:
                left = location.left()
                top = location.top()
                right = location.right() 
                bottom = location.bottom() 
                face= (top, right, bottom,  left)
                locations_list.append(face)
                face_encodings = face_recognition.face_encodings(frame[:, :, ::-1], locations_list)
                MAX_DISTANCE=0.6
                for face_encoding in face_encodings:
                    distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    # select the closest match (smallest distance) if it's below the threshold value
                    if np.any(distances <= MAX_DISTANCE):
                        best_match_idx = np.argmin(distances)
                        name = known_face_names[best_match_idx]
                    else:
                        name = None
                    paint_detected_face_on_image(frame, face, name)
        
        cv2.imshow('Video', frame)

            
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

database = setup_database()
run_face_recognition_video(database)

