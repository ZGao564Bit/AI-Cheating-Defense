import tensorflow as tf
import numpy as np
import cv2

interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()


im = cv2.imread("pubg_noise_02_02.png")
img = tf.image.resize_with_pad(np.expand_dims(im, axis=0), 192, 192)
input_image = tf.cast(img, dtype=tf.float32)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
interpreter.invoke()
keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
for j in range(7):
    print(keypoints_with_scores[0][0][j][2])



