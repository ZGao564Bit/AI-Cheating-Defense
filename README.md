> ## Objective:
>
> The objective of our product is defending the AI-based cheating tools. The cheating tools we are trying to defend are based on Human Pose Estimation. This is a computer vision technology which can detect human's body parts locations. Cheating tools can use this location information to automatically aim and shoot.
>
> The model which we assume that the tools may use is "movenet/singlepose" from TensorFlow hub. Due to we don't know the detail and content of this model, we can only use black box method to defend. Which means we have to calculate the gradient of the model numerally.

## Programming Language:

Python 3.7

## Library and Model:

Main Libraries: OpenCV, TensorFlow, pyautogui.

Model: movenet/multipose/lightning.

<https://tfhub.dev/google/movenet/multipose/lightning/1>


## System Environment Requirements:

Python 3.7 or later.

#### **Packages requirements:**

For the cheating tool:

TensorFlow, OpenCV-Python, pyautogui.

For the perturbation generating program:

TensorFlow, OpenCV.

## Operations:

**Cheating tool:** Run the [GameShooting.py](http://GameShooting.py) in PyCharm. Or in command line with Administrator privilege.

**Perturbation generating:** Run [addPoints.py](http://addPoints.py) in the first iteration. If some of the body parts do not have the confidence value lower than 0.4, run [Optimize.py](http://Optimize.py) with the input of the perturbed image output from [addPoitns.py](http://addPoitns.py).

## Special instructions:

1. For the perturbation generating programs ([addPoints.py](http://addPoints.py) and [Optimize.py](http://Optimize.py)), The format of input images must be png or other lossless formats supported by OpenCV.
2. The perturbation generation is highly time consumptive, usually takes several hours. The program will print the confidence values after each time each pixel changed.
3. The perturbation generation can not deal with the local-minimum problem. If you run the [Optimize.py](http://Optimize.py) for several times and still can not get all body parts' confidence values lower than 0.4, it means you meet the local-minimum problem.
4. A txt file with information about the changes to pixels will be generated after program exiting. It includes: 1. The corresponding body parts. 2. The GBR (Green Blue Red) values of the changed pixels.
5. The perturbation generating program can be accelerated by GPUs with CUDA.
