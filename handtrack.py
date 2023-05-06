import cv2
import mediapipe as mp
import numpy as np
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import torch
import time
import math

#program works best with a flat hand to point at objects due to AI reading hand better for depth -- works best when it sees wrist
#works best closer 

#TODO:remove log 10 from z calc
#Async work on average logic for loc depth and such for Parkinson's pointer feature 
#TODO: figure out that crash  File "c:\CodeRepos\BradikenisticPointer\BradikenisticPointer\handtrack.py", line 58, in getZfromDepth
        #return self.currDepth[cy][cx]
        #IndexError: index 384 is out of bounds for axis 0 with size 384
#TODO: may need to otherwise add a buffer: CANNOT HIT IF Z IS +-10 of starting Z 
#TODO: adjust param of model --> to make it less polar from 255->150 and 150->0, since this is an ISSUE!!!! 
#TODO: add the average for parkinsons
#may need to adjust hand tracker walker -- to get depth before applying AI hands to do walker on

class handTracker():

    def __init__(self, mode=False, maxHands=1, detectionCon=0.6,modelComplexity=1,trackCon=0.6):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
        #variables for position and directional vector of fingers
        self.nailPosX = -1 #hand 8
        self.nailPosY = -1 #hand 8
        self.nailPosZ = -1 

        self.wristX = -1 #hand 5 #body of finger  TODO: USING BODY OF FINGER more than wrist now
        self.wristY = -1 #hand 5 
        self.wristZ = -1 

        self.fingerDirVecX = -1
        self.fingerDirVecY = -1
        self.fingerDirVecZ = -1 
    
        self.currDepth = []
        self.currImage = []
        self.feature_extractor = None
        self.model = None
        self.device = None

    def getWidthOfCurrImage(self):
        return int(len(self.currImage[0]))
    def getHeightOfCurrImage(self):
        return int(len(self.currImage))
    def getRatioYX(self):
        return self.getHeightOfCurrImage()/self.getWidthOfCurrImage()

    def getZfromDepth(self, cx, cy): # higher z is closer to cam 
        if(cy >= int(len(self.currDepth)) or cx>=int(len(self.currDepth[int(cy)]))):
            return -1
        else:
            #print(len(self.currDepth[cy]))
            return self.currDepth[int(cy)][int(cx)]

    def DrawFailCode(self):
        cv2.putText(self.currImage, "No Hit", (int(self.getWidthOfCurrImage()/2), int(self.getHeightOfCurrImage()/2)), 1, 1,(255,0,0), 2, cv2.LINE_AA)

    def walkTillHit(self):
        cx = self.nailPosX
        cy = self.nailPosY
        cz = self.nailPosZ
        #in body
        while(True):
            if(cx<0 or cx >= self.getWidthOfCurrImage()):
                print("failed X")
                return None
            elif(cy<0 or cy >= self.getHeightOfCurrImage()):
                print("failed Y")
                return None
            elif(cz < self.getZfromDepth(cx,cy) and cz < self.nailPosZ-(abs(self.nailPosZ-self.wristZ))):
                cv2.circle(self.currImage,(int(cx),int(cy)), 10 , (255,50,0), cv2.FILLED) #HIT OBJECT, DO STUFF
                return 1
            else:
                cx+=(self.fingerDirVecX)
                cy+=(self.fingerDirVecY)
                cz+=(-1*(abs(self.fingerDirVecZ))) #incase you have hand over predicted hand location -- need -
                


    def GetDepthNew(self): #REMEMBER: run on GPU AS LONG AS POSSABLE, conevert to cpu minimal --   torch.cuda.synchronize() to test if gpu slow since .cpu() is a sync point -- use manual         torch.cuda.synchronize()
        pixel_values = self.feature_extractor(self.currImage, return_tensors="pt").pixel_values
        with torch.no_grad(): 
            outputs = self.model(pixel_values.to(self.device)) 
            predicted_depth = outputs.predicted_depth
        # interpolate to original size
        prediction = torch.nn.functional.interpolate( # is expression does not run
                            predicted_depth.unsqueeze(1),
                            size= (self.getHeightOfCurrImage(),self.getWidthOfCurrImage()), 
                            #scale_factor = ,
                            mode="bicubic",
                            align_corners=False,
                    ).squeeze()
        outputGPU = prediction.cuda()
        #print("a")
        output = outputGPU.detach().cpu().numpy() 
        #print("b")
        formatted = (output * 255 / np.max(output)).astype('uint8')
        #print("c")
        return formatted

    def handsFinder(self,image,draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
                    
        return cv2.resize(image, (int(self.getWidthOfCurrImage()),int(self.getHeightOfCurrImage())), interpolation = cv2.INTER_AREA)
    
    def positionFinder(self,image, handNo=0, draw=True):
        foundNail = False
        foundBase = False
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark): #we only need pos 8, and 7-5 avg
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                if(cx>=self.getWidthOfCurrImage() or cy >= self.getHeightOfCurrImage()): #Making sure not off right side of screen
                    return
                if id == 8:
                    self.nailPosX = cx
                    self.nailPosY = cy
                    self.nailPosZ = self.getZfromDepth(cx,cy)
                    if draw:
                        cv2.circle(image,(cx,cy), 15 , (255,0,255), cv2.FILLED)
                    foundNail = True
                
                elif id == 6: 
                    self.wristX = cx
                    self.wristY = cy
                    self.wristZ = self.getZfromDepth(cx,cy)
                    if draw:
                        cv2.circle(image,(cx,cy), 15 , (255,0,255), cv2.FILLED)
                    foundBase = True
                    
            
            self.fingerDirVecX = (self.nailPosX - self.wristX)/10
            self.fingerDirVecY = (self.nailPosY - self.wristY)/10
            self.fingerDirVecZ = (self.nailPosZ - self.wristZ)/10
            if(self.fingerDirVecZ>10):
                self.fingerDirVecZ= self.fingerDirVecZ/10
            #length = (self.fingerDirVecX**2+self.fingerDirVecY**2+self.fingerDirVecZ**2)**0.5
            #self.fingerDirVecX = self.fingerDirVecX/length  
            #self.fingerDirVecY = self.fingerDirVecY/length 
            #self.fingerDirVecZ = self.fingerDirVecZ/length 
            
            print((self.fingerDirVecX,self.fingerDirVecY,self.fingerDirVecZ))
            
            #if foundBase == False:
            #    self.wristX = -1
            #    self.wristY = -1
            #if foundNail == False:
            #    self.nailPosX = -1
            #    self.nailPosY = -1 
                





def main():
    #Launch video capture from standard position of computer camera.
    cap = cv2.VideoCapture(0)
    tracker = handTracker()
    tracker.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    tracker.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)    
    tracker.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(tracker.device)
    tracker.model.to(tracker.device)
    
    #Hold camera on while window is running.
    #Note: Previous image is tracked to do the depth disparity.
    #Calculate current 'grey' earlier in doing so. (self.gray* variables to do it)
    while True:
        success,tracker.currImage = cap.read()
        tracker.currImage = tracker.handsFinder(tracker.currImage)
        
        #depth getter code
        tracker.currDepth = tracker.GetDepthNew()

        tracker.positionFinder(tracker.currImage)
        
        if tracker.nailPosX == -1 or tracker.wristX == -1:
            tracker.DrawFailCode()
        elif tracker.walkTillHit() == None:
            tracker.DrawFailCode()

        cv2.imshow("Video",tracker.currImage)
        
        cv2.imshow("Depth", tracker.currDepth)
        
        cv2.waitKey(1)

if __name__ == "__main__":
    main()