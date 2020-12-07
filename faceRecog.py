# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:50:15 2020

@author: berkant tuğberk
"""
#Importing libraries
import cv2
import face_recognition
import numpy as np
import os

#The function for finding encodings of the faces
def findEncodings(images):
    encodeList=[]
    
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList



path= "C:/Users/tugberk/Desktop/python/faceRecog/images"
images = []
names=[]
mylist = os.listdir(path)
#Parse the file and find the encodings of the face. If you upload more images in images file, the image will be recognized.
for list in mylist:
    currentImage=cv2.imread(f'{path}/{list}')
    images.append(currentImage)
    names.append(os.path.splitext(list)[0])

encodeListKnown = findEncodings(images)
print("Encoding is completed...")


capture = cv2.VideoCapture(0)

while True:
    success,img = capture.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)
    
    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex=np.argmin(faceDis)
        
        if matches[matchIndex]:
            name= names[matchIndex].upper()
            print(name)
            
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
