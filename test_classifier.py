import sys
sys.path.append('..')
import ObjectDetection.core as ObjectDetection
import ImageClassification.core as ImageClassification
# import ImageClassification.Hashing.core as Hashing
from ObjectTracker.OpticalFlowAPI.core import  OpticalFlowMultiTracker
import cv2
import numpy as np
import time
import datetime
from ImageClassification.Squeezenet.core import SqueezeNetClassifier

def hamming_diff(a, b):
    return a - b

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
#cap = cv2.VideoCapture(0)
detector = ObjectDetection.Detector()
classifier = ImageClassification.Classifier()
alt_classifier = SqueezeNetClassifier()
# hash_classifier = Hashing.Classifier()
tracker = OpticalFlowMultiTracker()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
result_video= cv2.VideoWriter('Test_classifier.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

count=0
detected_count=0
while True:
    ret, frame = cap.read()
    # frame=cv2.flip(frame,0)
    # frame=cv2.flip(frame,+1)
    resolution = (640, 480)
    frame = cv2.resize(frame, resolution)
    print(frame.shape)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints = tracker.computeKeyPoints(frame)
    print('Keypoints ', len(keypoints))
    if len(keypoints) > 100:
        res, frame_out = detector.perform_detection(frame.copy())
        if not len(res) == 0:
            for boxes in res:
                print(boxes)
                box = boxes[2]
                label = boxes[0]
                score = boxes[1]
                sx, sy, ex, ey = box
                window = frame[sy:ey, sx:ex]
                print('Box', box)
                print(window.shape)
                result = classifier.perform_classification(window)
                alt_results = alt_classifier.predict(window)
#                 hash_result = hash_classifier.predict(window)
#                 print('Hash_result', hash_result)
                pred = result[-1]
                alt_pred = alt_results[-1]
                print(label,score)
                #print('Peleenet preds', result)
                #print('Squeezenet preds', alt_results)
                #print("peleenet: ", pred)
                #print("squeezenet: ", alt_pred)
#                 cv2.putText(frame, hash_result['hash_class'], (int(sx), int(sy + 20)), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
#                cv2.putText(frame, str(pred[0]) + '|' + pred[1], (int(sx-100), int(sy - 200)), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0))
                cv2.putText(frame, str(pred[0]) + '|' + pred[1], (int(10), int(90)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
                cv2.putText(frame, str(alt_pred[0]) + '|' + alt_pred[1], (int(10), int(120)), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0))
                cv2.rectangle(frame, (sx, sy), (ex, ey), (255, 255, 255), 5)
                cv2.putText(frame, str(score) + '|' + str(label), (int(10), int(60)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255 ))
                cv2.putText(frame, str(count) +'-'+str(detected_count), (int(10), int(30)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0 ))
                detected_count+=1
    cv2.putText(frame, str(count), (int(10), int(30)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0 ))
    count+=1
    result_video.write(frame)
    cv2.imshow('classification', frame)
    a = cv2.waitKey(1)
    if a == 27:
        break
