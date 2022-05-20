import os
import cv2 as cv
import numpy as np


haar_cascade=cv.CascadeClassifier('haar_face.xml')
DIR=r'C:\Users\Mohamed Sameh\Desktop\Practice\Python\OpenCV\Face Detection\People'

people=[]
for i in os.listdir(DIR):
   people.append(i)

#features=np.load('features.npy',allow_pickle=True)
#labels=np.load('labels.npy')

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img=cv.imread('cr7.jpg')
img=cv.resize(img,(500,500))#only for tutorial purpose (size is too big)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

#cv.imshow('Person',gray)

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

for (x, y, w, h) in faces_rect:
   faces_roi=gray[y:y+h,x:x+h] #faces region of interest/crop out the faces in the image
   label,confidence=face_recognizer.predict(faces_roi)
   print(f'{people[label]} with confidence={confidence}')

   cv.putText(img,str(people[label]),(20,40),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=3)
   cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=3)

cv.imshow('Face Recognition',img)

cv.waitKey(0)