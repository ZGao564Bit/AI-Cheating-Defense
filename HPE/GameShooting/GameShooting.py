import pyautogui
import tensorflow as tf
import numpy as np
import datetime
from PIL import ImageGrab
from matplotlib import pyplot as plt
import cv2
import win32api
import pydirectinput
import keyboard

interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()
#f = open('keylogger.txt', 'w')
#f.write('True')
#f.close()
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        #print(kp)
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)
        #     print("[" + str(ky) + " " + str(kx) + " " + str(kp_conf) + "]    ok")
        # else:
        #     print("[" + str(ky) + " " + str(kx) + " " + str(kp_conf) + "]")

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))


    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            if (p1 == 5 and p2 == 6) or (p1 == 6 and p2 == 5):
                ym = (y1 + y2) / 2 + 440
                xm = (x1 + x2) / 2 + 860
                pydirectinput.moveTo(int(xm), int(ym), duration=1/30)
                pydirectinput.click()
                #win32api.SetCursorPos((int(xm), int(ym)))
                #pyautogui.dragTo(int(xm), int(ym), duration=1 / 30)
                #pyautogui.click()


#name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# get the size of screen and used to adjust
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
# = cv2.VideoWriter('test_%s.avi'%name, fourcc, 20, (1920, 1080))
#video = cv2.VideoWriter('test_%s.avi'%name, fourcc, 5, (640, 480))
#video = cv2.VideoWriter('test_%s.avi'%name, fourcc, 5, (368, 368))
#video = cv2.VideoWriter('test_%s.avi'%name, fourcc, 5, (1920, 1080))
#video = cv2.VideoWriter('test_%s.avi'%name, fourcc, 5, (200, 200))

while True:
    #if open('keylogger.txt', 'r').read() == 'False':
        #break
    # print(line)
    #im = ImageGrab.grab((640, 300, 1280, 780))
    #im = ImageGrab.grab()
    #im = ImageGrab.grab((776, 356, 1144, 724))
    im = ImageGrab.grab((860, 440, 1060, 640))
    im.save('path-to-save.png')
    img = cv2.imread(r'path-to-save.png')
    # print(type(img))
    frame = img
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    #print(keypoints_with_scores)

    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)

    #cv2.imshow('Movenet Lightning', frame)
    #imm = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #video.write(imm)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

#cap.release()
#cv2.destroyAllWindows()
#video.release()

right_eye = keypoints_with_scores[0][0][2]
left_elbow = keypoints_with_scores[0][0][7]
shaped = np.squeeze(np.multiply(interpreter.get_tensor(interpreter.get_output_details()[0]['index']), [480, 640, 1]))

for kp in shaped:
    ky, kx, kp_conf = kp
    print(int(ky), int(kx), kp_conf)



