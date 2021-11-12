from PIL import ImageGrab
import numpy as np
import cv2
import datetime
from pynput import keyboard
import threading

#flag of ceasing recording
flag=False

#record process
def video_record():
    #get current time
    name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    #get current screen
    p = ImageGrab.grab()
    #get the size of screen and used to adjust
    a, b = p.size
    #encoding format
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter('test_%s.mp4'%name, fourcc, 20, (a, b))
    while True:
        im = ImageGrab.grab()
        #convert RGB to BGR
        imm=cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        video.write(imm)
        if flag:
            print("Record end！")
            break
    video.release()

#Keyboard listener, press esc to cease recording
def on_press(key):
    global flag
    if key == keyboard.Key.esc:
        flag=True
        return False

#main process
#Create a thread to record
th=threading.Thread(target=video_record)
th.start()
print("Start recording")

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()