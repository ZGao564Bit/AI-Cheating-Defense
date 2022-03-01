# About our Project
Machine learning can do a lot kind of work. Recently in the vedio game community, there is a kind of cheating programs use machine learning. Because they are different than the traditional gameing cheating programs, they are hard for game companies to detect and prevent. The goal of our project is developing a method to defend this kind of programs. 

There are two of main tasks in our project: 
1. Developing a ML based cheating software: 
    1) The program is writtien in Python
    2) Based on Human Pose Estimation
    3) Uses the trained model: MobileNetV2 lightning model
    4) Auto mouse moving and clicking

2. Finding a method to defend.  
    1) Adversarial Examples

## Cheating Program: Operation Steps for our program 
1. run "keyboardlistener.py"
2. run "detect.py"
3. open the picture/video/game
4. click on "ctrl+shift+alt", the analysis will begin
5. click on "ctrl+alt+z", the analysis will stop.

## Methods to defend: Adversarial Examples 
1. Used to condidence value to find the places and colors which will have best effects to confuse the machine learning program. (backup method)
2. Used to fast gradient sign method (FGSM) to add adversarial examples to the game images. 

## Methods to defend: Operation Steps
1. TBD

# Project Tasks Assignment
Project manager: **Zichun Gao**     
**Chang Liu, Yue Sun:** Cheating program development.   
**Zhenyu Liao:** Backup method to generate adversarial examples.    
**Zichun Gao:** Generating adversarial examples using FGSM.     
