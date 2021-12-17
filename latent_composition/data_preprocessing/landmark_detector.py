
import cv2
import numpy as np
import pandas as pd
import urllib.request as urlreq

import os
import matplotlib.pyplot as plt
from pylab import rcParams

def download_models():
     # Downloading haar cascade face detector model 
    haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    haarcascade = "haarcascade_frontalface_alt2.xml"

    # chech if file is in working directory
    if (haarcascade in os.listdir(os.curdir)):
        # print("File exists")
        temp = 404
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml, < 1MB
        urlreq.urlretrieve(haarcascade_url, haarcascade)
        print("File downloaded")

    # Downloading the face landmark detector model 
    LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
    LBFmodel = "lbfmodel.yaml"

    # check if file is in working directory
    if (LBFmodel in os.listdir(os.curdir)):
        # print("File exists")
        temp = 404 
    else:
        # download file from url and save locally as lbfmodel.yaml, < 54MB
        urlreq.urlretrieve(LBFmodel_url, LBFmodel)
        print("File downloaded")

    return haarcascade, LBFmodel

# Extract landmark points for the given image path 
def extract_landmarks(img_path, detector, landmark_detector):
    img = cv2.imread(img_path) 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) 

    # Initialize face detector - create an instance of the Face Detection Cascade Classifier
    faces = detector.detectMultiScale(img_gray)
    if (faces == ()):
        return None

    face = faces[:1,:] # Selecting only the first detected face for processing 
    
    # Detect landmarks on "image_gray"
    _, landmarks = landmark_detector.fit(img_gray, face)

    for landmark in landmarks:
        for x,y in landmark[0]:
            # display landmarks on "image_cropped"
            # with white colour in BGR and thickness 1
            cv2.circle(img_gray, (int(x), int(y)), 1, (255, 255, 255), 1)
    return landmarks


# Computing the facial landmarks using the off-the-shelf library to estimate landmarks dlib/opencv
def compute_facial_landmarks(src_img_folder, ldmk_save_path): 
    haarcascade, LBFmodel = download_models() 

    # Initialize face detector - create an instance of the Face Detection Cascade Classifier
    detector = cv2.CascadeClassifier(haarcascade)
    
    # Initialize landmark detector - create an instance of the Facial landmark Detector with the model
    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(LBFmodel) 

    columns = [str(i//2) for i in range(0, 68*2)]
    for id in range(0, len(columns)):
        if (id%2 == 0):
            columns[id] += '_x'
        else: 
            columns[id] += '_y'

    columns = ['file_name'] + columns
    landmarks_list = []
    idx = 0
    for img_nm in os.listdir(src_img_folder):
        img_path = os.path.join(src_img_folder, img_nm)
        
        landmarks = extract_landmarks(img_path, detector, landmark_detector)   
        
        # If faces has been detected and landmarks are computed then only run this loop 
        if (landmarks):
            landmarks_flatten = [img_nm]
            
            for landmark in landmarks:
                for x,y in landmark[0]:
                    landmarks_flatten.extend([x, y])

            landmarks_list.append(landmarks_flatten)
        
        # if (idx > 10):
        #     break
        idx += 1
        if (idx%100 == 1):
            print("landmark detection done for: ", idx)

    landmarks_list = np.array(landmarks_list)
    landmark_df = pd.DataFrame(data = landmarks_list, columns = columns)
    print("landmarks_df shape: ", landmark_df.shape)
    landmark_df.to_csv(ldmk_save_path)
    print("Saving landmark files to: ", ldmk_save_path) 


        
def main():
    src_img_folder = '../CelebAMask-HQ/data_filtered/img' 
    ldmrk_save_path = '../data_files/landmarks_detected.csv'
    compute_facial_landmarks(src_img_folder, ldmrk_save_path)

if __name__ == "__main__":
    main()