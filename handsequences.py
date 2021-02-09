import cv2
import sys
sys.path.append('..')

#from HandDetector import Detector
from handdetectionscripts.HandDetector import Detector

#/home/reorder/Documents/Krithika_handdetection/handdetectionscripts/new_hand_weight/latest/yolov3-tiny_obj.cfg

videoPath = sys.argv[1]

outvideo=videoPath.split('.')
output = outvideo[0]+'_hand'+'.avi'
print(output)

cap = cv2.VideoCapture(videoPath)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

result_video= cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

detector = Detector()
count=0
while True:
	ret,frame=cap.read()
	if (ret==False):
		break
	#resolution = (640, 480)
	#frame = cv2.resize(frame, resolution)
	img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	res, frame_out = detector.perform_detection(frame.copy())
	for bbox in res:
		label = bbox[0]
		score = bbox[1]
		sx, sy, ex, ey = bbox[2]
		cv2.rectangle(frame, (sx, sy), (ex, ey), (255, 255, 255), 5)
		cv2.putText(frame, str(score) + '|' + str(label), (int(10), int(60)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255 ))
	cv2.putText(frame, str(count), (int(10), int(30)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0 ))
	count+=1
	result_video.write(frame)
	cv2.imshow('classification', frame)
	a = cv2.waitKey(100)
	if a == 27:
		break
