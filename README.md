 # Point-DPT
The first (known) opensource pointer program which tracks the position of a user's index finger to find out where they are pointing on the screen WITHOUT external depth cameras/sensors.

## How are Pipeline Works
### Tracking the Finger
Using the mediapipe python package we are able to track a hand and get the following information:
![image](https://user-images.githubusercontent.com/72413722/236637614-cba2e540-579e-45f0-a3cf-101ff3a53bf0.png)
Using the position of the tip of the index finger and a point on the body of the finger, we create a direction vector to determine where the finger is pointing.
### Creating a Depth Map
We use [Intel's dpt-hybrid-midas model](https://huggingface.co/Intel/dpt-hybrid-midas) to approximate a depth map for the current frame. The depth map information is used to find the Z coordinate of the direction vector.
### Determining Position of Finger Point
We used the direction vector to perform ray marching to find where the finger is pointing. Starting with the position of the tip of the index finger(nail), we aplly the direction vector iteravily until one of these condtions are true [these are simplified slightly as modifications are made to improve accuracy, such as finding distances which are invalid and faults of the AI system providing garbage data]:
1. x/y location of ray shot out is outside bounds of visible frame (fails to draw
2. z location of ray is behind xy location of depth image -- like the senario below 
3. not satisfied 1. or 2. so we add X & Y & Z direction vector to our current ray position
![image](https://user-images.githubusercontent.com/67922228/236638892-2f4be85d-e4e6-4f4f-ab78-f00ebf738d94.png)

### Recommended Hardware (for real time processing): 
- Jetson Orin Nano (embedded AI hardware makes this a substantially cheaper option to use) 
or 
- RTX 3060 

### Future use-cases 
assuming performance issues are not of notiable issue -- since camrea images can easily be sent as a low resolutuon to be processed by an off short server to remove client overhead of using the AI model 
- very intuitive UX/UI without special depth reading hardware
- useful for placement of furniture using finger gestures
- powerful for mapping hand movments of patients with poor control of hand mobility
- enables pointing out objects of intrest when words are not avaible for an individual to speak -- specifically allowing better text to speech tools by pointing at blocks of words
- empowers hardware with less capabilities to run AR/VR applications which use depth data

### How To Build:
Clone the repo:<br />
```$ git clone https://github.com/16-Bit-Dog/point-dpt```<br />
Install requirments by running the following command in the point-dpt directory<br />
```$ pip install -r requirments.txt```

