# Denne klassen inneholder all funksjonalitet for ansiktsgjenkjenning
# Brukes i fcrecog-sss.py via OOP

import face_recognition
import cv2
import os
import glob
import numpy as np

class FaceRecognizer:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for raskere prosessering
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):

        # Importering bildene
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Lagring av bildene som har blitt kodet
        for img_path in images_path:
            img = cv2.imread(img_path)
            if img is None:
                print("Kunne ikke lese bilde: {}".format(img_path))
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Hent filnavnet fra hele filstien
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)

            # Hent encoding - sjekk at ansikt ble funnet
            encodings = face_recognition.face_encodings(rgb_img)
            if len(encodings) == 0:
                print("Ingen ansikt funnet i: {}".format(img_path))
                continue

            img_encoding = encodings[0]

            # Lagre filnavn og encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Finn alle ansikter i nåværende frame
        # Konverter fra BGR (OpenCV) til RGB (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Sjekk om ansiktet matcher kjente ansikter
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Bruk ansiktet med minst avstand (best match)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Konverter til numpy array for å justere koordinater med frame resizing
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
