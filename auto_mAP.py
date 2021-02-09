import glob
import os

folder = "/home/reorder/Documents/Hand_Darknet_Model/Hand_Detector/Version_02/Weights_Negative"
iou = 0.80
#-iou_thresh 0.60

log_file = open("mAP_log.txt", "a+")

for file in glob.glob(folder + '/*.weights'):
	print("weights currently checking :",file)
	log_file.write("weights currently checking :" + file +"\n"+"\n")
	os.chdir('/home/reorder/darknet/')
	if iou ==0.50 :
		command='./darknet detector map /home/reorder/Documents/Hand_Darknet_Model/Hand_Detector/Version_02/hand.data /home/reorder/Documents/Hand_Darknet_Model/Hand_Detector/Version_02/yolov3-tiny_obj.cfg ' + file
	else :
		command='./darknet detector map /home/reorder/Documents/Hand_Darknet_Model/Hand_Detector/Version_02/hand.data /home/reorder/Documents/Hand_Darknet_Model/Hand_Detector/Version_02/yolov3-tiny_obj.cfg ' + file + ' -iou_thresh ' + str(iou)
	#os.system(command)
	print(command)
	log_file.write("Command:" + command + "\n"+"\n")
	return_back = os.popen(command).read()
	log_file.write("return_results:" + str(return_back[500:1500]) + "\n")
	#log_file.write("return_results:" + str(return_back[2350:3000]) + "\n")
	log_file.write("_______________________________________________"+"\n")
	log_file.flush()

