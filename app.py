#import prebuilt Packages
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'
import cv2
import streamlit as st
from PIL import Image
st.title("CAR NUMBER PLATE DETECTION AND RECOGNITION")
#taking image input
img = st.sidebar.file_uploader("Choose an image")
custom_config = r'--psm 10--oem 3'
if img is not None:
    img = Image.open(img)
    st.image(img,caption = "Uploaded Image")
    if st.button("PREDICT"):
        img = np.array(img)
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray_img = cv2.bilateralFilter(gray_img,15,90,90)
        NUM_PLATE_MODEL = cv2.CascadeClassifier("/content/haarcascade_russian_plate_number.xml")
        NUM_PLATE_DEMO = NUM_PLATE_MODEL.detectMultiScale(img,1.1,10)
        flag = 0 
        for (x,y,w,h) in NUM_PLATE_DEMO:
            flag = 1
            cv2.rectangle(img,(x,y),(x+w,y+h),[255,0,0],4)
            st.title("NUMBER PLATE DETECTED ")
            st.image(img)
            roi = gray_img[y:y+h,x:x+w]
            img = img[y:y+h,x:x+w]
            # Finding text from image using OCR
            text = pytesseract.image_to_string(roi,config = custom_config)
            st.title(" EXTRACTED NUMBER PLATE ")
            st.image(img)
            st.title(f" Extracted Number : {text}")
            # For finding each individual Chracter 
            Boxes = pytesseract.image_to_boxes(roi)
            hImg,wImg,d = img.shape
            for b in Boxes.splitlines():
                b = b.split()
                x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
                cv2.rectangle(img,(x,hImg-y),(w,hImg-h),(0,0,255),2)
            st.title(" EXTRACTED NUMBER PLATE with Characters  ")
            st.image(img)
        else :
            if flag == 0:
                st.title(" NO Number Plate Detected")
