#Populate the arry based on the configuration file 
#Do the conversion 


# Importing the required libraries...

import cv2 
import os 
import sys
import numpy as np 
import matplotlib.pyplot as plt

print("Python3 prg.py Input_Floder config_anno_file_path Output_folder")
print (len(sys.argv))

#Save the input parametes from comamnd line - sys.argv[0]
#check the value is equal 4  else you need to quit the program 

#Check the config files is available //  os.path.exists(file_path)

#open the config file // open 	
#copy the contents to an array -- for_loop

#Open the annotation folder 
#Retrive only name from complete path_file name// for_loop
#Read the first Text_file.. 
#Read the respective_Image ..
#Store the file name a variable -- 
#calculate width and hight .... -- shape()
#crop the image ...

#save it in the output folder with same name...

if len(sys.argv) >= 4 :
	print('Input_Floder',sys.argv[1])
	print('config_Annotation_txt',sys.argv[2])
	#print('IPAnnotation_folder',sys.argv[3])
	print('OP_Dir',sys.argv[3])
	Input_Floder= sys.argv[1]
	if not os.path.exists(Input_Floder):
		print("Input_Floder file missing")
		sys.exit(1)
	config_anno_file_path= sys.argv[2]
	if not os.path.isfile(config_anno_file_path):
		print ("config file missing")
		sys.exit(1)
	Output_folder= sys.argv[3]
	if not os.path.exists(Output_folder):
		print ("Output_folder is missing")
		sys.exit(1) 
	config_text_file= open (config_anno_file_path,'r')
	config_Text = config_text_file.readlines()
	for f in config_Text:
		Image_name= f[:f.rfind('.')]
		text = open(Image_name+'.txt')
		Text_data=text.readlines()
		abs_fname=Image_name[Image_name.rfind('/')+1:]
		img_file = cv2.imread(Image_name+'.jpg')
		for i in Text_data:
			Text_content=i.split(',')
			Text_content[-1] = Text_content[-1].strip()
			x1  = float(Text_content[0])
			y1  = float(Text_content[1])
			x2  = float(Text_content[2])
			y2  = float(Text_content[3])
			x3  = float(Text_content[4])
			y3  = float(Text_content[5])
			x4  = float(Text_content[6])
			y4  = float(Text_content[7])
			TEXT  = (Text_content[8])
			cenx  = min(x1,x4)
			ceny  = min(y1,y2)
			cenw  = max(x2,x3)
			cenh  = max(y3,y4)
			crop_width = int(cenw - cenx)
			crop_height= int(cenh - ceny)
			x=int(cenx)
			y=int(ceny)
			ex=int(crop_width)
			ey=int(crop_height)
			cr_img = img_file[y:y+ey,x:x+ex]
			Name = abs_fname+TEXT
			cv2.imwrite(Output_folder+'/'+Name+'.jpg',cr_img)
			TExt_inside = Output_folder+'/'+Name+'.jpg'+'	'+TEXT
			output_File = open("Output_txt",'a+')
			output_File.write(TExt_inside+'\n')






















			
