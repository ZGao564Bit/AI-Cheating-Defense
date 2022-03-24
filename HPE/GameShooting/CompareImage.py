import tensorflow as tf
import numpy as np
np.set_printoptions(suppress=True)
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import random
from scipy.signal import convolve, convolve2d

# img_path1 = 'test1.jpg'
# img_path2 = 'output1.jpg'
#
# frame1 = cv2.imread(img_path1)
# frame2 = cv2.imread(img_path2)
#
# img = frame1 - frame2


def getShape(img):
    frame = img
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    y = frame.shape[0]
    x = frame.shape[1]

    shaped = np.squeeze(np.multiply(keypoints_with_scores, [y, x, 1]))
    return shaped

def isEqual(img):
    for i in range(300):
        for j in range(300):
            for k in range(3):
                if img[i][j][k] != 0:
                    return False
    return True

img_path = 'test2.png'
img = cv2.imread(img_path)
cv2.imwrite('com1.png', img)
img1 = cv2.imread('com1.png')

interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()
res1 = getShape(img)
res2 = getShape(img1)
print(res1)
print(res1-res2)
comp = img - img1
print(isEqual(comp))
#print(img)