# Point-DPT
Pointer program which tracks the position of a user's index finger to find out where they are pointing on the screen.

## How are Pipeline Works
### Tracking the Finger
Using the mediapipe python package we are able to track a hand and get the following information:
![image](https://user-images.githubusercontent.com/72413722/236637614-cba2e540-579e-45f0-a3cf-101ff3a53bf0.png)
Using the position of the tip of the index finger and a point on the body of the finger, we create a direction vector to determine where the finger is pointing.
### Creating a Depth Map
We use [Intel's dpt-hybrid-midas model](https://huggingface.co/Intel/dpt-hybrid-midas) to approximate a depth map for the current frame. The depth map information is used to find the Z coordinate of the direction vector.
### Determining Position of Finger Point
We used the direction vector to perform ray marching to find where the finger is pointing. Starting with the position of the tip of the index finger(nail), we aplly the direction vector iteravily until 
