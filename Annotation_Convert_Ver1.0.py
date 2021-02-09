#Check the input,config file is available
#Populate the arry based on the configuration file 
#Do the conversion 


# Importing the required libraries...

import cv2 
import os 
import sys 

print("Python3 prg.py class.names IPAnnotationtext OPAnnotatiofolder")
print (len(sys.argv))

#Save the input parametes from comamnd line - sys.argv[0]
#check the value is equal 4  else you need to quit the program 

#Check the config files is available //  os.path.exists(file_path)

#check the annotation folder is available // os.path.isfile(file_path) 

#open the config file // open 	
#copy the contents to an array -- while () -- array[count++]  

#Open the annotation folder // while - os.listdir(folder)
#Read the first file 
#Store the file name a variable --  
#Stroe the full file name with OPAnnotatiofolder+"/"+file name
#spit the contents based on space - split(,") 
#replace the first vlue to the value of array -    
#assing to a variable array[2] Con_str_First file_content[2] file_content[3] file_content[4] file_content[5] Con_Str_Second
#sys.command(echo "above variable > newoutput file name)
#Move to the next file name 

if len(sys.argv) >= 4 :
	print('Names_file',sys.argv[1])
	print('Annotation_txt',sys.argv[2])
	#print('IPAnnotation_folder',sys.argv[3])
	print('OP_Dir',sys.argv[3])
	class_name= sys.argv[1]
	if not os.path.isfile(class_name):
		print("Class_name file missing")
		sys.exit(1)
	IPAnnotationtextpath= sys.argv[2]
	if not os.path.isfile(IPAnnotationtextpath):
		print ("config file missing")
		sys.exit(1)
	OPAnnotationfolder= sys.argv[3]
	if not os.path.exists(OPAnnotationfolder):
		print ("OPAnnotation folder is missing")
		sys.exit(1)
	class_file = open (class_name,'r')
	class_data = class_file.readlines()
	config_txt= open (IPAnnotationtextpath,'r')
	config_data = config_txt.readlines()
	Con_str_First = " 0 0 0 "
	Con_Str_Second = " 0 0 0 0 0 0 0 "
	for f in config_data:
		Image_name= f[:f.rfind('.')]
		text = open(Image_name+'.txt')
		Text_data=text.read()
		Text_line=Text_data.split('\n')
		abs_fname=Image_name[Image_name.rfind('/')+1:]
		out_train = open(OPAnnotationfolder+'/'+abs_fname+'.txt', "w+")
		print("Type of file content")
		print(type(Text_line))		
		print("Content of file content")
		print(Text_line)		

		file_Content = ""
		for i in Text_line:
			if len(i) <= 5:
				continue
			print("Content of the file")
			print(type(i))		
			print("Value of a line ")
			print(i)		
			Text_content=i.split()
			print("Class vlue")
			print(Text_content[0])		
			temp_classname = class_data[int(Text_content[0])].split('\n')
			#temp_classname.replace("\n",'')

			temp_store = temp_classname[0]+","+Con_str_First+","+Text_content[1]+","+Text_content[2]+","+Text_content[3]+","+Text_content[4]+","+Con_Str_Second+"\n";
			file_Content +=temp_store

		out_train.write(file_Content)
		out_train.close() 
		#text.close()	  

