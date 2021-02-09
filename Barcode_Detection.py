import time
import cv2
import tqdm
import os
from skimage import morphology
import numpy as np
import time

start_1 = time.time()
folder = '/home/reorder/Documents/Basic_ImageProcessing/Barcode_comp/barcode_horizontal'
#folder = '/home/sunil/Documents/Advik/Basic_ImageProcessing/x'
for filename in os.listdir(folder):
		
    start = time.time()
    #brgimg = cv2.imread(filename)
    #im_rgb = cv2.cvtColor(brgimg, cv2.COLOR_BGR2RGB)
    Image_name= filename[:filename.rfind('.')]
    im_rgb = cv2.imread(os.path.join(folder,filename))
    resized_image = cv2.resize(im_rgb, (480,640))
    #print(type(resized_image))
    #cv2.imshow('RGB',resized_image)
    
    grayimg=cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    #cv2.imshow('GRAY',grayimg)
    median = cv2.medianBlur(grayimg,5)
    #cv2.imshow('MEDIAN',median)

    SE = cv2.getStructuringElement(cv2.MORPH_RECT,(32,3))
    tophat = cv2.morphologyEx(median, cv2.MORPH_TOPHAT,SE)
    #cv2.imshow('TOPHAT_SE',tophat)
    
    (thresh, im_bw) = cv2.threshold(tophat, 220, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #print(thresh)

    SE_2 = cv2.getStructuringElement(cv2.MORPH_RECT,(4,2))
    opening = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, SE_2)
    #cv2.imshow('MP_open',opening)
    
    SE_3 = cv2.getStructuringElement(cv2.MORPH_RECT,(11,3))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, SE_3)
    #cv2.imshow('MP_close_fill',closing)

    contours=cv2.findContours(closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    for cnt in contours:
        Im_Fill=cv2.drawContours(closing, [cnt],0,255,-1)
    
    SE_4 = cv2.getStructuringElement(cv2.MORPH_RECT,(8,4))
    dilation = cv2.dilate(Im_Fill,SE_4,iterations = 1)
    #cv2.imshow('Dilation',dilation)

    dilation = dilation >220
    cleaned = morphology.remove_small_objects(dilation, min_size=6000, connectivity=1)

    cleaned = cleaned.astype(int)
    cleaned=np.uint8(cleaned*255)

    contours=cv2.findContours(cleaned,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    for contour in contours:
        if cv2.contourArea(contour) < 2048: 
           continue
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        #print(aspect_ratio)
        if aspect_ratio < 0.3:
           continue
        # making green rectangle arround the moving object 
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255,0), 5)
        crop_img = resized_image[y:y+h, x:x+w]

    #cv2.imshow('Text_Detected_Box',resized_image)

    #(x, y, w, h) = cv2.boundingRect(contours[0])        
    #crop_img = resized_image[y:y+h, x:x+w]
    #cv2.imshow('Text_detected',crop_img)
    #cv2.imwrite(folder+'/'+Image_name+'_0'+'.jpg',resized_image)
  
    cv2.waitKey(1)
    End = time.time()
    Time = End-start
    #print(Time)
eND =time.time()
#print("Time_2",eND-start_1)
print("Barcode_detection.py am done")
