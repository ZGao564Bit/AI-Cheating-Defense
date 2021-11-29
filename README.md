##About our program
Our program is designed to be able to find human heads in the screen and automatically move the mouse. 

It runs with the following steps and rules:
1. Automatically take screenshots and save pictures in real time 
2. Call the lightweight openpose model for human posture recognition, the main method is to obtain the key points of human bones through openpose (Directly use the DNN Library of OpenCV to call the Tensorflow model of openpose, and then post process the output)
3. Extract and distinguish the key points (human joints) from the 18 feature maps which the system got from the model.
4. Automatically move the mouse

##Operation Steps:
1. run "keyboardlistener.py"
2. run "detect.py"
3. open the picture/video/game
4. click on "ctrl+shift+alt", the analysis will begin
5. click on "ctrl+alt+z", the analysis will stop.

