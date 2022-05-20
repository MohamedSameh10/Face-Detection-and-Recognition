import os
import cv2 as cv
import numpy as np


haar_cascade=cv.CascadeClassifier('haar_face.xml')
DIR=r'C:\Users\Mohamed Sameh\Desktop\Practice\Python\OpenCV\Face Detection\People'
"""
In Python, backslash is used to signify special characters.
For example, "hello\nworld" -- the \n means a newline. Try printing it.
Path names on Windows tend to have backslashes in them. But we want them to mean actual backslashes, not special characters.
r stands for "raw" and will cause backslashes in the string to be interpreted as actual backslashes rather than special characters.
"""
features=[]
labels=[]

people=[]
for i in os.listdir(DIR):
   people.append(i)

""" 
listdir() returns a list containing the names of the entries in the directory given by path. The list is in arbitrary order.
It does not include the special entries '.' and '..' even if they are present in the directory
"""

def create_train():
    #loop on folders (people) containing images
    for person in people:
        path=os.path.join(DIR,person)
        label=people.index(person)

        #loop on images inside the folder
        for img in os.listdir(path):
            img_path=os.path.join(path,img)

            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces_rect:
                faces_roi=gray[y:y+h,x:x+h] #faces region of interest/crop out the faces in the image
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done......')

features=np.array(features,dtype='object')
labels=np.array(labels)

face_recognizer=cv.face.LBPHFaceRecognizer_create()
#Train the recognizer on the features list and the labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
"""NPY file is a NumPy array file created by the Python software package.
The format stores all of the shape and data type information necessary to reconstruct the array correctly even on another machine with a different architecture."""
np.save('features.npy',features)
np.save('labels.npy',labels)

