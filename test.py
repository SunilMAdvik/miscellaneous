import cv2
import sys
import os
import numpy as np

videoPath = sys.argv[1]
print("V: ",videoPath )
cap = cv2.VideoCapture(videoPath)

# so, convert them from float to integer. 
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 

result = cv2.VideoWriter('filename.avi',  
                         cv2.VideoWriter_fourcc(*'XVID'), 
                         10, size) 
                         
                         

while(True): 
    ret, frame = cap.read() 
  
    if ret == True:  
  
        # Write the frame into the 
        # file 'filename.avi' 
        result.write(frame) 
  
        # Display the frame 
        # saved in the file 
        cv2.imshow('Frame', frame) 
  
        # Press S on keyboard  
        # to stop the process 
        if cv2.waitKey(1) & 0xFF == ord('s'): 
            break
  
    # Break the loop 
    else: 
        break
  
# When everything done, release  
# the video capture and video  
# write objects 
cap.release() 
result.release() 
    
# Closes all the frames 
cv2.destroyAllWindows() 
   
print("The video was successfully saved") 
