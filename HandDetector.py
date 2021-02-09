import cv2
# import tensorflow as tf
import numpy as np
import os
import sys
import time
import argparse
import importlib
sys.path.extend(['../../'])

#import ObjectDetection.DarknetAPI.core as DarknetCore
import ReorderHandler.ObjectDetection.DarknetAPI.experimental.core as DarknetCore
'''
refresh_params:
    task:
        loads the detector requested in params db

perform_detection:
    task:
        performs inference request for a particular image with the loaded detector
/home/reorder/reordercontroller/ReorderHandler/HandDetection/new_hand_weight/yolov3-tiny_obj.cfg

'''
class Detector():
    def __init__(self):
        self.supported_models=['yolov3','tiny-yolo']
        self.refresh_params()
    def refresh_params(self):
        self.params=dict()
        self.params['yolov3_use_api']='darknet'
        self.params['darknet_model_type']='tiny_yolo'
        self.params['yolov3_current_weights_location']='/home/reorder/rfid/reordercontroller/ReorderHandler/HandDetection/new_hand_weight/latest/yolov3-tiny_obj_33000.weights'
        self.params['yolov3_current_cfg_location']='/home/reorder/rfid/reordercontroller/ReorderHandler/HandDetection/new_hand_weight/latest/yolov3-tiny_obj.cfg'
        #self.params['yolov3_current_cfg_location']='/home/reorder/Documents/Krithika_handdetection/handdetectionscripts/new_hand_weight/latest/yolov3-tiny_obj.cfg'

        self.params['darknet_names_root']='/home/reorder/rfid/reordercontroller/ReorderHandler/HandDetection/new_hand_weight/hand_c.names'
        self.num_classes=1
        self.params['yolov3_current_meta_path']='/home/reorder/rfid/reordercontroller/ReorderHandler/HandDetection/new_hand_weight/hand.data'
        if self.params['darknet_model_type']=='yolov3' or self.params['darknet_model_type']=='tiny_yolo':
            self.yolov3_weights=self.params['yolov3_current_weights_location']
            self.yolov3_configuration=self.params['yolov3_current_cfg_location']
            self.yolov3_names=self.params['darknet_names_root']
            self.num_classes=1
            self.yolov3_meta_path=self.params['yolov3_current_meta_path']
            if 'yolov3_use_api' in self.params:
                if self.params['yolov3_use_api']=='darknet':
                    importlib.reload(DarknetCore)
                    # from DarknetCore import DarknetDetector
                    self.model={
                        'type':'yolov3',
                        'model':DarknetCore.DarknetDetector(self.params)
                        }
    def perform_detection(self,image):
        if self.model is None:
            print('Model not loaded yet!')
            return
        return self.model['model'].performDetect(image,self.params)
'''def send_data(data):
    pass
def listen(port):
    context=zmq.Context()
    socket=context.socket(zmq.SUB)
    socket.connect('tcp://localhost:'+str(port))
    socket.setsockopt(zmq.SUBSCRIBE,b'PROC_FRAME')
    sleep(2)
    detector=Detector()
    while True:
        topic=socket.recv_string()
        frame=socket.recv_pyobj()
        res,frame=detector.perform_detection(frame)
if __name__='__main__':
    parser=argparse.ArgumentParser('Object Detection API for Reorder')
    parser.add_argument('--port',type=int,default=5555)
    args=parser.parse_args()
    listen(args.port)
    # not sure what arguments can be sent
'''
