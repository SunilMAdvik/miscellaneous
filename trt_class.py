import os
import pycuda.autoinit

from HandDetection.Hand_trt.yolo_with_plugins import TrtYOLO

class Detector():
    def __init__(self):
        #trt_yolo = TrtYOLO(args.model, (h, w), args.category_num)
        self.trt_yolo = TrtYOLO('yolov3-tiny-416', (416,416),1)
	#img = cv2.imread('/home/reorder/frame_5.jpeg')
    def perform_detection(self,image):
        if self.trt_yolo is None:
           print('Model not loaded yet!')
           return
        boxes, confs, clss = self.trt_yolo.detect(image, 0.3)
        return boxes,image
