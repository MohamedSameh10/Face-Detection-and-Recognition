#https://github.com/opencv/opencv/tree/master/data/haarcascades
import cv2 as cv

img=cv.imread('saf.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

haar_cascade=cv.CascadeClassifier('haar_face.xml')
faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10) #output= array (x,y,w,h) w=h because it's a square

for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)

print(faces_rect)
cv.imshow('Face Detection',img)
cv.waitKey(0)