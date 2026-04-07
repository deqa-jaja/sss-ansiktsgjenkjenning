#!/usr/bin/python3

#importing the necessary packages (libraries)-importering de nødvendie pakkene

import subprocess
import cv2
import os
import time
import glob
from face_recognizer import FaceRecognizer
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play




#importering bilder fra images

sfr = FaceRecognizer()
sfr.load_encoding_images("images/")


# funksjon 1
#  - kamera oppstart

def Camera_start(width, height):
    global camera_process
    cmd = [
        "libcamera-vid", "-t", "0",
        "--segment", "1", "--codec", "mjpeg", "-n",
        "-o", "/run/shm/test%06d.jpg",
        "--width", str(width), "--height", str(height)
    ]
    camera_process = subprocess.Popen(cmd, preexec_fn=os.setsid)


# stopp kamera prosessen
def Camera_stop():
    try:
        os.killpg(os.getpgid(camera_process.pid), 15)
    except Exception:
        pass


#  definering av variablene (initialise variables)
width        = 720
height       = 540
last_greeted = ""

os.makedirs("sounds", exist_ok=True)

cv2.namedWindow('Frame')
Text = "Left Mouse click on picture to EXIT, Right Mouse click for eye detaction ON/OFF"
ttrat = time.time()

Camera_start(width,height)



try:
    while True:

        if time.time() - ttrat > 3 and ttrat > 0:
            Text =""
            ttrat = 0

        # importering bildene
        pics = glob.glob('/run/shm/test*.jpg')
        while len(pics) < 2:
            pics = glob.glob('/run/shm/test*.jpg')
            time.sleep(0.01)
        pics.sort(reverse=True)

        img = cv2.imread(pics[1])
        if img is None:
            continue

        # slett gamle bilder
        if len(pics) > 2:
            for tt in range(2,len(pics)):
                try:
                    os.remove(pics[tt])
                except OSError:
                    pass

        # putting rectangles around detected faces-Markering ansikter 
        face_locations, face_names = sfr.detect_known_faces(img)
        for face_loc, name in zip(face_locations, face_names):
             y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

             # sett navne på bildene
             cv2.putText(img, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 200), 4)

             # skape en lydfil basert på kundens ansikt
             if name != "Unknown" and name != last_greeted:
                 try:
                     tts = gTTS("velkommen kjære kunde " + name, lang='no')
                     tts.save("sounds/hello.mp3")
                     song = AudioSegment.from_mp3("sounds/hello.mp3")
                     play(song)
                     last_greeted = name
                 except Exception:
                     pass

        
        # fremvisning av bildene, med rektangler rundt ansiktene & navnene på ansiktene
        cv2.putText(img,Text, (10, height - 10), 0, 0.4, (0, 255, 255))
        cv2.imshow('Frame', img)

        # Avslutt med 'q' tast
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

# rydder opp
Camera_stop()
cv2.destroyAllWindows()
