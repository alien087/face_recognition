#!/usr/bin/env python
# coding: utf-8

# In[12]:


import cv2
from modules.predict_face import encoding, predict, load_encode


# In[11]:


# Get a reference to webcam 
video_capture = cv2.VideoCapture(0)
#Function to load encode file
encode_path = 'model/encoding.pickle'
data = load_encode(encode_path)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color 
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, encodings = encoding(image)  
    names = predict(encodings, data)
    
    if(names is not None):
        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)

   
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()


# In[ ]:




