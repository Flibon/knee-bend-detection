import cv2
import numpy as np
import time
import PoseModule as pm

period=time.time()
period1=8.1
t1=0
flag=int
flag2=int  
count=0
cap = cv2.VideoCapture(0)#r"C:\Users\Manav\OneDrive\Desktop\Manav\Python\intern\KneeBendVideo.mp4")
detector=pm.poseDetector()
x=int   
feedback=""
while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    img = detector.findPose(img,False)
    lmList = detector.findPosition(img, False)
    if len(lmList) != 0:
        x=detector.findAngle(img,23,25,29)      #Right leg
        #detector.findAngle(img,24,26,28)      #Left leg
    
    t=time.time()
    s=time.time()
    if x<140:
        t=time.time()
        if t-t1>=period1:
            t1=time.time()
        if t-t1>=8:
            flag=1
        if t-t1<8:
            if flag==0:
                flag2=1
        feedback=""
    else:
        t=0
        t1=0 
        if flag==1:
            count+=1
            flag=0
        if flag2==1:
            flag2=0
            feedback="Keep your knee bent."

            
       
    performance=(count/(s-period))*100
    if performance>=4:
        r="You are doing great."
    elif 1<performance<4:
        r="Increase your speed!"
    elif performance<1:
        r="Lets start."
    
    cv2.putText(img,feedback, (10,60),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.putText(img,"Rep="+str(count), (10,20),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(img,"Stats", (800,20),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(img,"Successful rep ="+str(count), (800,60),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(img,"Time passed ="+str(int(s-period))+ " sec", (800,90),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(img, "Performance = "+str('%.2f'%performance), (800,150),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(img,"Calories burned ="+str(0.123*count)+ " cal", (800,120),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(img,r, (800,190),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
   
    cv2.imshow("Image",img)
    cv2.waitKey(1)