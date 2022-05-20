import cv2 as cv

img=cv.imread('cr7.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

haar_cascade=cv.CascadeClassifier('haar_eye.xml')
eye1_rect,eye2_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
for (x,y,w,h) in eye1_rect,eye2_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)

cv.imshow('eye detector',img)

cv.waitKey(0)