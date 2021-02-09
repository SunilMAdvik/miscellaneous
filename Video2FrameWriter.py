import cv2
import sys
import os
import numpy as np

videoPath = sys.argv[1]
print("V: ",videoPath )
cap = cv2.VideoCapture(videoPath)
outPath = videoPath.replace(".avi", "")
if(not os.path.exists(outPath)):
    os.makedirs(outPath)
    
print(cap.get(3),cap.get(4),cap.get(6),cap.get(7))
i = 1
crop_value = 0.27
crop_row = int(crop_value * 416)
fast_feature_detector = cv2.FastFeatureDetector_create(25)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
#outvideo = cv2.VideoWriter('out.avi', fourcc, 10.0, (416, 416))
while True:
    ret, frame = cap.read()
    if(ret == False):
        break
    fileName = "frame_{0}.jpg".format(i)
    filePath = os.path.join(outPath, fileName)
    fCrop = frame[crop_row:, :]
    gray = cv2.cvtColor(fCrop, cv2.COLOR_BGR2GRAY)
    kp = fast_feature_detector.detect(gray, None)
    keypoint_ = len(kp)
    out = frame.copy()
#     out[crop_row:, :] = cv2.drawKeypoints(out[crop_row:, :], kp, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.line(out, (0, crop_row), (frame.shape[1], crop_row), [0, 0, 0], 5)
    cv2.imwrite(filePath, out)
    print("writing ", fileName)
    i += 1
    

cap.release()
