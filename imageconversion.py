import cv2 
import os 
import sys
import glob
import numpy as np 

folder = '/home/reorder/Documents/Hand_Darknet_Model/2021_Handdataset_annotated'
output_folder = '/home/reorder/Documents/Hand_Darknet_Model/2021_Handdataset_annotated'

for file in glob.glob(folder + '/*.jpeg'):
	img_file = cv2.imread(file)
	Image_name= file[:file.rfind('.')]
	abs_fname=Image_name[Image_name.rfind('/')+1:]
	cv2.imshow("window_name", img_file)
	cv2.waitKey(10)
	cv2.imwrite(output_folder+'/'+abs_fname+'.jpg',img_file)
