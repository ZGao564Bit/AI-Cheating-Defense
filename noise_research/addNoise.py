import os
import cv2
import numpy as np
import random

def sp_noise(noise_img, proportion):
    height, width = noise_img.shape[0], noise_img.shape[1]
    num = int(height * width * proportion)
    for i in range(num):
        w = random.randint(0, width - 1)
        h = random.randint(0, height - 1)
        if random.randint(0, 1) == 0:
            noise_img[h, w] = 0
        else:
            noise_img[h, w] = 255
    return noise_img

def gaussian_noise(img, mean, sigma):
    img = img / 255
    noise = np.random.normal(mean, sigma, img.shape)
    gaussian_out = img + noise
    gaussian_out = np.clip(gaussian_out, 0, 1)
    gaussian_out = np.uint8(gaussian_out*255)
    return gaussian_out

def random_noise(image,noise_num):
    img_noise = image
    rows, cols, chn = img_noise.shape
    for i in range(noise_num):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img_noise[x, y, :] = 255
    return img_noise

def convert(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        path = input_dir + "/" + filename
        print("doing... ", path)
        noise_img = cv2.imread(path)
        img_noise = gaussian_noise(noise_img, 0, 0.12)
        # img_noise = sp_noise(noise_img,0.025)
        #img_noise  = random_noise(noise_img,500)
        cv2.imwrite(output_dir+'/'+filename,img_noise )


if __name__ == '__main__':
    input_dir = "C:/Users"
    output_dir = "C:/Users"
    convert(input_dir, output_dir)