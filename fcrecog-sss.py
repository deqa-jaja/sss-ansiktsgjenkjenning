#!/usr/bin/python3

# Ansiktsgjenkjenning og velkomstsystem for Raspberry Pi
# Bruker kamera til å gjenkjenne kunder og hilse dem med lyd

import subprocess
import numpy as np
import cv2
import os
import time
import glob
from face_recognizer import FaceRecognizer
import gtts
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play


# Importering av kjente ansikter fra images-mappen
sfr = FaceRecognizer()
sfr.load_encoding_images("images/")


# Kamera oppstart - starter libcamera på Raspberry Pi
def Camera_start(width, height):
    global camera_process
    cmd = [
        "libcamera-vid", "-t", "0",
        "--segment", "1", "--codec", "mjpeg", "-n",
        "-o", "/run/shm/test%06d.jpg",
        "--width", str(width), "--height", str(height)
    ]
    camera_process = subprocess.Popen(cmd, preexec_fn=os.setsid)


# Kamera stopp - rydder opp prosessen
def Camera_stop():
    try:
        os.killpg(os.getpgid(camera_process.pid), 15)
    except Exception:
        pass


# Definering av variabler
width  = 720
height = 540
# Bruk haarcascade som følger med OpenCV
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

check = 1
face_detected = 0
last_greeted = ""  # holder styr på siste person som ble hilst på

cv2.namedWindow('Frame')
Text = "Left Mouse click on picture to EXIT, Right Mouse click for eye detaction ON/OFF"
text_timer = time.time()

Camera_start(width, height)


try:
    while True:

        # Fjern velkomstteksten etter 3 sekunder
        if time.time() - text_timer > 3 and text_timer > 0:
            Text = ""
            text_timer = 0

        # Importering av bilder fra kamera
        pics = glob.glob('/run/shm/test*.jpg')
        while len(pics) < 2:
            pics = glob.glob('/run/shm/test*.jpg')
            time.sleep(0.01)
        pics.sort(reverse=True)

        img = cv2.imread(pics[1])
        if img is None:
            continue

        # Slett gamle bilder for å spare minne
        if len(pics) > 2:
            for old_pic in range(2, len(pics)):
                try:
                    os.remove(pics[old_pic])
                except OSError:
                    pass

        # Markering av ansikter med rektangler og navn
        if check == 1:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            face_detected = 0

            face_locations, face_names = sfr.detect_known_faces(img)
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                # Sett navn på ansiktene
                cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 200), 4)

                # Spill velkomstlyd kun for nye gjenkjenninger
                if name != "Unknown" and name != last_greeted:
                    try:
                        tts = gtts.gTTS("velkommen kjære kunde " + name, lang='no')
                        tts.save("sounds/hello.mp3")
                        song = AudioSegment.from_mp3("sounds/hello.mp3")
                        play(song)
                        last_greeted = name
                    except Exception:
                        pass

        # Fremvisning av bildene med rektangler og navn
        cv2.putText(img, Text, (10, height - 10), 0, 0.4, (0, 255, 255))
        cv2.imshow('Frame', img)

        # Avslutt med 'q' tast
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

# Rydder opp
Camera_stop()
cv2.destroyAllWindows()
