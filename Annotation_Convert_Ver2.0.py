#Check the input,config file is available
#Populate the arry based on the configuration file 
#Do the conversion 


# Importing the required libraries...

import cv2 
import os 
import sys
import numpy as np 
import matplotlib.pyplot as plt

print("Python3 prg.py class.names IPAnnotationtext OPAnnotatiofolder")
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
	print('Names_file',sys.argv[1])
	print('config_Annotation_txt',sys.argv[2])
	#print('IPAnnotation_folder',sys.argv[3])
	print('OP_Dir',sys.argv[3])
	class_name= sys.argv[1]
	if not os.path.isfile(class_name):
		print("Class_name file missing")
		sys.exit(1)
	config_anno_file_path= sys.argv[2]
	if not os.path.isfile(config_anno_file_path):
		print ("config file missing")
		sys.exit(1)
	OPAnnotationfolder= sys.argv[3]
	if not os.path.exists(OPAnnotationfolder):
		print ("OPAnnotation folder is missing")
		sys.exit(1) 
	config_text_file= open (config_anno_file_path,'r')
	config_Text = config_text_file.readlines()
	for f in config_Text:
		Image_name= f[:f.rfind('.')]
		text = open(Image_name+'.txt')
		Text_data=text.read()
		Text_line=Text_data.split('\n')
		abs_fname=Image_name[Image_name.rfind('/')+1:]
		#out_train = open(OPAnnotationfolder+'/'+abs_fname+'.jpg', "w+")
		for i in Text_line:
			Text_content=i.split()
			print(Text_content)			
			if len(i) <= 5:
				continue		
			img_file = cv2.imread(Image_name+'.jpg')
			#img1= cv2.cvtColor(img_file ,cv2.COLOR_BGR2RGB)
			height = int(img_file.shape[0])
			width  = int(img_file.shape[1])
			cenx=float(Text_content[1])
			ceny=float(Text_content[2])
			crop_width=float(Text_content[3])
			crop_height=float(Text_content[4])
			cenx =int(cenx* width)
			ceny=int(ceny* height)
			crop_width=int(crop_width*width)
			crop_height=int(crop_height*height)
			x=int(cenx-crop_width/2)
			y=int(ceny-crop_height/2)
			ex=int(x+crop_width)
			ey=int(y+crop_height)
			crop_img = img_file[y:ey,x:ex]
			cv2.imwrite(OPAnnotationfolder+'/'+abs_fname+'.jpg',crop_img)
		#out_train.close() 



