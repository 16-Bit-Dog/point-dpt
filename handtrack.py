import cv2
import mediapipe as mp
import numpy as np
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import torch
import time

#TODO: we need Z axis, depth of finger nail, and body, avg to get final component of vector
\
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
        self.fingerDirVecX = 0
        self.fingerDirVecY = 0 
        self.wristX = 0 #hand 0 
        self.wristY = 0 #hand 0 
        self.currDepth = []
        self.currImage = []
        self.feature_extractor = None
        self.model = None
        self.device = None

    def GetDepthNew(self): #REMEMBER: run on GPU AS LONG AS POSSABLE, conevert to cpu minimal
        print("1")
        pixel_values = self.feature_extractor(self.currImage, return_tensors="pt").pixel_values
        with torch.no_grad(): 
            print("a")
            outputs = self.model(pixel_values.to(self.device)) 
            print("b")
            predicted_depth = outputs.predicted_depth
        # interpolate to original size
        prediction = torch.nn.functional.interpolate( # is expression does not run
                            predicted_depth.unsqueeze(1),
                            size= (int(len(self.currImage)),int(len(self.currImage[0]))), 
                            #scale_factor = ,
                            mode="bilinear",
                            align_corners=False,
                    ).squeeze()
        print("c")
        outputGPU = prediction.cuda()
        print("2")
        output = outputGPU.detach().cpu().numpy() #run  prediction on gpu, and move to cpu tensor, then convert to numpy
        #output = prediction.cpu().numpy() # rn slow
        print("d")
        formatted = (output * 255 / np.max(output)).astype('uint8')
        print("e")
        return formatted

#    def ConvertToGreyScale(self, image): #Converting the background to grey, managing depth 
#        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#    def GetDepth(self):
#        
#        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=255)
#        disparity = stereo.compute(self.prevGrey, self.currGrey)
#        cv2.imshow("depth",disparity)

    def handsFinder(self,image,draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
                    
        return cv2.resize(image, (int(len(image[0])*0.8),int(len(image)*0.8)), interpolation = cv2.INTER_AREA)
    
    def positionFinder(self,image, handNo=0, draw=True):
        
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark): #we only need pos 8, and 7-5 avg
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                if id == 8:
                    self.nailPosX = cx
                    self.nailPosY = cy
                
                elif id == 0: #body of finger
                    self.wristX = cx
                    self.wristY = cy
                        
                    self.fingerDirVecX = self.nailPosX - self.wristX
                    self.fingerDirVecY = self.nailPosX - self.wristY
                        
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
    tracker.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
    tracker.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    
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