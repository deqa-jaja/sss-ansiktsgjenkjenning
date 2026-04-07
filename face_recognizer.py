# Denne klassen inneholder funksjonaliteten for ansiktsgjenkjenning
#Brukes i fcrecog-sss.py via OOP metode


import face_recognition
import cv2
import os
import glob
import numpy as np

class FaceRecognizer:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # resize frame for a faster speed
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):

        # importering bildene-Downloading images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # lagring bildene som har blitt kodet-Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            if img is None:
                print("Kunne ikke lese bilde: {}".format(img_path))
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # get the filename only from the initial file path
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # get encoding 
            # - sjekk at ansikt ble funnet
            encodings = face_recognition.face_encodings(rgb_img)
            if len(encodings) == 0:
                print("Ingen ansikt funnet i: {}".format(img_path))
                continue

            img_encoding = encodings[0]

            # Store file name og file encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # find all the faces and face encodings in the current frame of video
        # convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            
            # see if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
