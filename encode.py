#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import matplotlib.pyplot as plt


# In[ ]:


data_path = 'dataset/'


# In[ ]:


#Save all image path
image_path = list(paths.list_images(data_path))

encoded = []
name_encoded = []


# In[ ]:


for (i, path) in enumerate(image_path):
    #Extract person name from the image path
    print(f"[INFO] Processing image {i+1}/{len(image_path)}")
    name = path.split(os.path.sep)[-2]
    
    #Load the train image and convert it to RGB
    #Since the default format from cv2 is BGR
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
    box = face_recognition.face_locations(img, model='hog')
    
    #After we find the boundix boxes, we encode the face
    encode = face_recognition.face_encodings(img, box)
    
    #After we get the encoded image, we save the encoding result
    #and the name of the image
    for encoding in encode:
        #save the encoding result
        encoded.append(encoding)
        name_encoded.append(name)
    


# In[ ]:


#Save the encoding result using pickle
print("[INFO] Saving the encodings...")
data = {"encodings": encoded, "names": name_encoded}
f = open("model/encoding.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
print("[INFO] Saved")


# In[ ]:




