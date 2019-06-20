import numpy as np
import cv2
import random
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')  

cap = cv2.VideoCapture(0)
r=255
g=0
b=0
count = 0
roi_color = None
roi_eyes = None
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.1, 10)
    for (x,y,w,h) in faces: 
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)  
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = frame[y:y+h, x:x+w]
        #cv2.imwrite(".\output\image"+str(count)+".png",roi_color)
        #count += 1
        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,10)
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
            print(ex,ey,ex+ew,ey+eh)
            roi_eyes = roi_color[ey:ey+eh,ex:ex+ew]
            cv2.imwrite(".\output\image" + str(count) + ".png", roi_eyes)
            count += 1
    cv2.imshow("Full",frame)
    if roi_color is not None:
        cv2.imshow('Face',roi_color)
    if roi_eyes is not None:
        cv2.imshow('Eyes',roi_eyes)

    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()