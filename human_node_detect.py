import os

import cv2 as cv
import matplotlib.pyplot as plt
import pyautogui
from PIL import ImageGrab

# 设定监听键盘初始状态
f = open('keylogger.txt', 'w')
f.write('False')
f.close()

auto = True  # 1. determine if cursor has moved to the position of head
os.system("mkdir output")  # 创建output目录


# create an img file [cv2.imwrite(filename, image)]
def write_out_file(outfilename, frame):
    cv.imwrite(outfilename, frame)


total_frame = 378

thr = 0.2
width = 368
height = 368

# Dictionary: assign body parts index.
BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], [
                  "RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], [
                  "Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

inWidth = width
inHeight = height

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

isSave = False


# Set time interval to 1 ms.
while cv.waitKey(1) < 0:
    flag = open('keylogger.txt', 'r').read()
    if flag == 'True':
        # Image capture
        im = ImageGrab.grab()
        im.save('path-to-save.png')
        img = cv.imread(r'path-to-save.png')
        # print(type(img))
        frame = img

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        # Mean subtraction and swap Red and Blue. reduce to half
        net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight),
                                          (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
        out = out[:, :19, :, :]

        assert (len(BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(BODY_PARTS)):
            heatMap = out[0, i, :, :]

            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > thr else None)

        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert (partFrom in BODY_PARTS)
            assert (partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                isSave = True
                print('Ok!!!')
                # mouse movement. Set movement time to 1 / 60 ms.
                pyautogui.moveTo(points[idFrom][0], points[idFrom][1], duration=1 / 60)
                # print(points[idFrom], points[idTo])
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, points[idFrom], (3, 3),
                           0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[idTo], (3, 3), 0,
                           0, 360, (0, 0, 255), cv.FILLED)

        t, _ = net.getPerfProfile()
        freq = cv.getTickFrequency() / 1000

        if isSave:
            write_out_file("ok.jpg", frame)
            write_out_file(".\\output\\" + 'filename_base' + "_" + str('index') + ".jpg", frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # plt.imshow(frame)
            # plt.show()
        isSave = False
    # break
