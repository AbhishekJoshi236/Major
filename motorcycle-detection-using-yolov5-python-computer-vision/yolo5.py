import os
import torch
import cv2
import numpy as np
from tracker import *

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('ROI')
cv2.setMouseCallback('ROI', POINTS)


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
count=0
cap=cv2.VideoCapture('tvid.mp4')
tracker=Tracker()

# new
# Create a directory to store cropped images
output_dir = 'rider_images'
os.makedirs(output_dir, exist_ok=True)
####
while True:
    ret,frame=cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    
    frame=cv2.resize(frame,(1020,600))
    
    
    results = model(frame)
    list=[]
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d=(row['name'])
        # print(d);
     
        if 'motorcycle' in d:
            ### Increase the height and width of the bounding box
            y1 = max(0, y1 - 20)  # Increase the top coordinate
            y2 = min(frame.shape[0], y2 + 20)  # Increase the bottom coordinate
            x1 = max(0, x1 - 20)  # Increase the left coordinate
            x2 = min(frame.shape[1], x2 + 20) 
            ###
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)        
            cv2.putText(frame,str(d),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            
            ###
            cropped_vehicle = frame[y1:y2, x1:x2]
            # Generate a unique filename for each cropped image
            image_filename = os.path.join(output_dir, f'cropped_vehicle_{count}.jpg')

            # Save the cropped image
            cv2.imwrite(image_filename, cropped_vehicle)
            ####
            
              
    # cv2.imshow("ROI",cropped_vehicle)
    cv2.imshow("ROI",frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
