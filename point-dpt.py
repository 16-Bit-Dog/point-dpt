import cv2
import mediapipe as mp
import numpy as np
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import torch
import time
import math

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
        
        self.cudaFound = False

        self.frameNum = 1
        self.AccumX = 0
        self.AccumY = 0

    def IncFrameNum(self):
        self.frameNum+=1
        if(self.frameNum == 12):
            self.frameNum = 1
            self.AccumX = 0
            self.AccumY = 0

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
            return self.currDepth[int(cy)][int(cx)]

    #draw fail text in middle of screen if pointer finds nothing by finding something off screen
    def DrawFailCode(self):
        cv2.putText(self.currImage, "No Hit", (int(self.getWidthOfCurrImage()/2), int(self.getHeightOfCurrImage()/2)), 1, 1,(255,0,0), 2, cv2.LINE_AA)

    def walkTillHit(self):
        cx = self.nailPosX
        cy = self.nailPosY
        cz = self.nailPosZ
        
        while(True):
            # Person is pointing off screen in the X direction
            if(cx<0 or cx >= self.getWidthOfCurrImage()):
                print("failed X")
                return None
            # Person is pointing off screen in the Y direction
            elif(cy<0 or cy >= self.getHeightOfCurrImage()):
                print("failed Y")
                return None
            #following code checks if your tracker position is behind the finger 
            elif(cz < self.getZfromDepth(cx,cy) and cz < self.nailPosZ-(abs(self.nailPosZ-self.wristZ+10))): #pushes nail backwards based on distance from base of finger to nail, this helps stop the moving 1 step and hitting same finger the tracker spawned from
                self.AccumX+=cx
                self.AccumY+=cy
                
                cv2.circle(self.currImage,(int(self.AccumX/self.frameNum),int(self.AccumY/self.frameNum)), 10 , (255,50,0), cv2.FILLED) #HIT OBJECT, DO STUFF
                return 1
            # Have not hit object yet, update current position
            else: 
                if(cz == 0): #if weird behavior from AI model
                    print("failed Z")
                    return None
                
                cx+=(self.fingerDirVecX)
                cy+=(self.fingerDirVecY)
                cz+=(-1*(abs(self.fingerDirVecZ))) #incase you have hand over predicted hand location -- need -
                


    def GetDepthNew(self): #REMEMBER: run on GPU AS LONG AS able, convert to cpu minimal -- torch.cuda.synchronize() to test if gpu slow since .cpu() is a sync point -- use manual         torch.cuda.synchronize()
        pixel_values = self.feature_extractor(self.currImage, return_tensors="pt").pixel_values
        with torch.no_grad(): 
            outputs = self.model(pixel_values.to(self.device))  #this is the slowest part of the code as profiled
            predicted_depth = outputs.predicted_depth 
        # interpolate to original size
        prediction = torch.nn.functional.interpolate( # is expression does not run
                            predicted_depth.unsqueeze(1),
                            size= (self.getHeightOfCurrImage(),self.getWidthOfCurrImage()), 
                            #scale_factor = ,
                            mode="bicubic",
                            align_corners=False,
                    ).squeeze()
        #allow non CUDA for debug builds, note: it is SLOW
        if self.cudaFound == True:
            outputGPU = prediction.cuda()
            output = outputGPU.detach().cpu().numpy() 
        else:
            output = predicted_depth.cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype('uint8')
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
            for id, lm in enumerate(Hand.landmark): #enumerate all hand objects found
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                if(cx>=self.getWidthOfCurrImage() or cy >= self.getHeightOfCurrImage()): #Making sure not off right side of screen
                    return
                
                #only work on nail and finger body -- id 8 & 6, if found draw a circle
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
                    
            #gets direction vector. 
            self.fingerDirVecX = (self.nailPosX - self.wristX)/10
            self.fingerDirVecY = (self.nailPosY - self.wristY)/10
            self.fingerDirVecZ = (self.nailPosZ - self.wristZ)/10
            #
            #if AI model bugs out then the dirvecZ will be over 100 and need pruning 
            if(self.fingerDirVecZ>10):
                self.fingerDirVecZ= self.fingerDirVecZ/10
            #commented out normalization code since this method is generally less accurate
            #length = (self.fingerDirVecX**2+self.fingerDirVecY**2+self.fingerDirVecZ**2)**0.5
            #self.fingerDirVecX = self.fingerDirVecX/length  
            #self.fingerDirVecY = self.fingerDirVecY/length 
            #self.fingerDirVecZ = self.fingerDirVecZ/length 


def main():
    #Launch video capture from standard position of computer camera.
    cap = cv2.VideoCapture(0)
    tracker = handTracker()
    
    #large model -- commented out since slower
    #tracker.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
    #tracker.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    #

    #lighter depth model
    tracker.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    tracker.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)
    #
    #get computational device -- prefer cuda
    if torch.cuda.is_available():
        tracker.device = torch.device("cuda:0")
        tracker.cudaFound = True
    else:
        tracker.device = torch.device("cpu")

    #set computational device
    tracker.model.to(tracker.device)
    
    #program main loop
    while True:
        tracker.IncFrameNum()
        success,tracker.currImage = cap.read()

        #uses model to find hand
        tracker.currImage = tracker.handsFinder(tracker.currImage)
        
        #depth image computer function call
        tracker.currDepth = tracker.GetDepthNew()

        #finds hand point positions
        tracker.positionFinder(tracker.currImage)
        
        #if invalid locations draw out of bound finger location-- else do image walker
        if tracker.nailPosX == -1 or tracker.wristX == -1:
            tracker.DrawFailCode()
        elif tracker.walkTillHit() == None:
            tracker.DrawFailCode()

        #draw images
        cv2.imshow("Video",tracker.currImage)
        cv2.imshow("Depth", tracker.currDepth)
        
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
