import cv2
import mediapipe as mp
import numpy as np
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import torch
import time

#program works best with a flat hand to point at objects due to AI reading hand better for depth

#TODO: may need before hand and after for depth reading

#Instantiation for the handTracker to hold tracking variables for video capture.
class handTracker():



    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
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
        self.nailPosX = 0 #hand 8
        self.nailPosY = 0 #hand 8
        self.nailPosZ = 0 

        self.wristX = 0 #hand 0 
        self.wristY = 0 #hand 0 
        self.wristZ = 0 

        self.fingerDirVecX = 0
        self.fingerDirVecY = 0 
        self.fingerDirVecY = 0 
        
        self.currDepth = []
        self.currImage = []
        self.feature_extractor = None
        self.model = None
        self.device = None

    def getWidthOfCurrImage(self):
        return int(len(self.currImage[0]))
    def getHeightOfCurrImage(self):
        return int(len(self.currImage))

    def getZfromDepth(self, x, y): # higher z is closer to cam 
        return self.currDepth[y][x]

    def walkTillHit(self):
        cx = self.nailPosX
        cy = self.nailPosY
        cz = self.nailPosZ
        #in body
        if(cx<0 or cx > self.getWidthOfCurrImage()):
            return None
        if(cy<0 or cy > self.getHeightOfCurrImage()):
            return None
        if(cz < self.getZfromDepth(cx,cy)):
            pass #HIT OBJECT, DO STUFF

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
                            mode="bilinear",
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
                    
        return cv2.resize(image, (int(self.getWidthOfCurrImage()*0.8),int(self.getHeightOfCurrImage()*0.8)), interpolation = cv2.INTER_AREA)
    
    def positionFinder(self,image, handNo=0, draw=True):
        
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark): #we only need pos 8, and 7-5 avg
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                if(cx>len(image[0]) or cy > len(image)): #Making sure not off right side of screen
                    break
                if id == 8:
                    self.nailPosX = cx
                    self.nailPosY = cy
                    
                    self.nailPosZ = self.getZfromDepth(cx,cy)
                
                elif id == 0: #body of finger
                    self.wristX = cx
                    self.wristY = cy
                    self.wristZ = self.getZfromDepth(cx,cy)
                    
                self.fingerDirVecX = self.nailPosX - self.wristX
                self.fingerDirVecY = self.nailPosX - self.wristY
                self.fingerDirVecZ = self.nailPosZ - self.wristZ

                print(self.nailPosZ)
                print(self.wristZ)
                print("\n")

                if id in [8, 0]:
                    #lmlist.append([id,cx,cy])
                    if draw:
                        cv2.circle(image,(cx,cy), 15 , (255,0,255), cv2.FILLED)

def depthGetter(tracker):
    #print(len(tracker.currImage[0]))
    #time.sleep(10)
    tracker.currDepth = tracker.GetDepthNew()

#        tracker.currGrey = tracker.ConvertToGreyScale(tracker.currImage)
#        if len(tracker.prevGrey) == 0:
#            tracker.prevGrey = tracker.currGrey 
#        tracker.GetDepth()

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
        depthGetter(tracker)
        
        tracker.positionFinder(tracker.currImage)
        
        cv2.imshow("Video",tracker.currImage)
        
        cv2.imshow("Depth", tracker.currDepth)
        
        cv2.waitKey(1)

if __name__ == "__main__":
    main()