# -*- coding: utf-8 -*-
"""
This project has been developed by
- Sai Pramod V.
- Sai Surya Teja T.
- Maanasa Gupta T.
- Gopi Krishna Reddy V.
- Srikanth M.
- Prudhvi D.
"""

import cv2
import imutils
import numpy as np
import keras
from keras.models import load_model
from PIL import ImageTk, Image
from keras.preprocessing import image
from skimage.transform import resize

model = load_model(r'C:/Users/Sai surya teja/new_model.h5')
model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])# global variables
bg = None

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    _,cnts,_= cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)
    
if __name__ == "__main__":
    aWeight = 0.5

    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 100, 350, 300, 590

    # initialize num of frames
    num_frames = 0
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                img = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
                test_image = resize(img,(128,128))
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis = 0)
                prediction = model.predict_classes(test_image)
                print(prediction)
                if(prediction[0]==0):
                    cv2.putText(clone, "Hi", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif(prediction[0]==1):
                    cv2.putText(clone, "one", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif(prediction[0]==2):
                    cv2.putText(clone, "super", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif(prediction[0]==3):
                    cv2.putText(clone, "victory", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Thesholded", thresholded)
                
                

        # draw the segmented handA
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypreAss by the user
        

        # if the user pressed "q", then stop looping
        if  cv2.waitKey(1) & 0xFF== ord("q"):
            camera .release()
            break 

cv2.destroyAllWindows()
