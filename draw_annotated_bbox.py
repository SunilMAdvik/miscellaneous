import cv2 
import os 
import sys
import glob
import numpy as np
from tqdm import tqdm 


folder = '/home/reorder/Documents/Hand_Darknet_Model/Failed_Datase/Hand'
output_folder = '/home/reorder/Documents/Hand_Darknet_Model/Failed_Datase/Hand/Hand_validate'
for file in tqdm(glob.glob(folder + '/*.jpg')):
	img_file = cv2.imread(file)
	#print(file)
	Image_name= file[:file.rfind('.')]
	#print("Image_name",Image_name)
	text = open(Image_name+'.txt')
	#print("Text_name",Image_name+'.txt')
	Text_data=text.readlines()
	#print("Text_data",Text_data)
	abs_fname=Image_name[Image_name.rfind('/')+1:]
	if len(Text_data) <1 :
		print(abs_fname)
	for i in Text_data:
		Text_content=i.split(',')
		#print("Text_content",Text_content)
		Text_content[-1] = Text_content[-1].strip()
		#print("Text_content",Text_content[-1])
		Text_content=Text_content[-1].split(' ')
		#print("Text_content",Text_content)
		shape = img_file.shape[0]
		Id  = float(Text_content[0])
		x  = float(Text_content[1])
		y  = float(Text_content[2])
		w  = float(Text_content[3])
		h  = float(Text_content[4])
		l = int((x - w / 2) * shape)
		r = int((x + w / 2) * shape)
		t = int((y - h / 2) * shape)
		b = int((y + h / 2) * shape)
		cv2.rectangle(img_file, (l, t), (r, b), (255, 255, 255), 3)
		Name = abs_fname
		cv2.imshow("window_name", img_file)
		cv2.waitKey(10)
		cv2.imwrite(output_folder+'/'+Name+'.jpg',img_file)


