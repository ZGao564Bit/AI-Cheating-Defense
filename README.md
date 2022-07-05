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
