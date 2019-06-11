import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
capture = cv2.VideoCapture(0)

cnt = 0
while True:
    # Capure frame-by-frame
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        print(x, y, w, h)
    #   capture your face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w] # (y, y + h)

        cv2.imwrite("images/Nikita/" + str(cnt) + ".png" , roi_gray)
        cnt +=1
        if cnt > 30:
            break

        color = (255, 0, 0) #BGR

        stroke = 2
        end_cord_x_width = x + w
        end_cord_y_height = y + h
        cv2.rectangle(frame,(x,y), (end_cord_x_width,end_cord_y_height), color, stroke)
    # Display images
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.release()
cv2.destroyAllWindows()
