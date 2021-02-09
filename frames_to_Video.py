import cv2
import os
import sys
sys.path.append('..')
import glob
import numpy as np
from tqdm import tqdm 


#/home/reorder/Documents/Krithika_handdetection/handdetectionscripts/new_hand_weight/latest/yolov3-tiny_obj.cfg

folder = '/home/reorder/Documents/Hand_Darknet_Model/Hand_Detector/test_dataset/Hand_test_data'


fourcc = cv2.VideoWriter_fourcc(*'XVID')
outvideo = cv2.VideoWriter('/home/reorder/Documents/Hand_Darknet_Model/Hand_Detector/Hand_video_2.avi', fourcc, 10.0, (416, 416))

for file in tqdm(glob.glob(folder + '/*.jpg')):
	frame = cv2.imread(file)
	cv2.imshow('classification', frame)
	out = frame.copy()
	outvideo.write(out)
	a = cv2.waitKey(100)
	if a == 27:
		break
