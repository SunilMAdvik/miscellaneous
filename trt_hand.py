import os
import pycuda.autoinit
import cv2
import sys
import os, psutil, gc, time, tracemalloc
sys.path.extend(['../'])
from HandDetection.Hand_trt.hand_trt import Detector
#from hand_trt import Detector

def main():
    tracemalloc.start()
#    snapshot = tracemalloc.take_snapshot()
    print("Start_ main",psutil.Process(os.getpid()).memory_info())
    start=time.time()
    detector = Detector()
    end=time.time()
    #detector= Detector()
    print("After_Detection_main",psutil.Process(os.getpid()).memory_info())
    print("HandDetection-trt Load time", end -start)
    #del detector
    #gc.collect()
    #print("After_Delete_main",psutil.Process(os.getpid()).memory_info())
    #exit(0)
    videoPath = sys.argv[1]
    outvideo=videoPath.split('.')
    output = outvideo[0]+'_hand'+'.avi'
    print(output)

    cap = cv2.VideoCapture(videoPath)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4)) 
    size = (frame_width, frame_height)

    result_video= cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
    print(psutil.Process(os.getpid()).memory_info())
    while True:
        ret,frame=cap.read()
        if (ret==False):
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print("Bfefore_allocation",psutil.Process(os.getpid()).memory_info())
        start=time.time()
        res, frame_out = detector.perform_detection(frame.copy())
        End = time.time()
        print("Ineference_Time_Trt",End-start)
        #print("After_allocation",psutil.Process(os.getpid()).memory_info())
        #print("Result Bbox", res)
#    print("End_ main",psutil.Process(os.getpid()).memory_info())
#    snapshot_end = tracemalloc.take_snapshot()
#    top_stats = snapshot_end.compare_to(snapshot, 'lineno')
#    for i in top_stats:
#        print(i)
if __name__== "__main__":
    main()

    
