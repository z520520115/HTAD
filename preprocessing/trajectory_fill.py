import numpy as np
import cv2
import os

file_root = '../simple_dataset/images/train/000003/'#当前文件夹下的所有图片
file_list = os.listdir(file_root)
save_out = '../simple_dataset/trajectory_mask/'#保存图片的文件夹名称

for img_name in file_list:
    if img_name.endswith('.jpg'):
        img_path = file_root + img_name
        imgs = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # h, w = imgs.shape[:2]
        mask = np.zeros((1280, 720), np.uint8)
        mask[] = 255

        # out_name = img_name.split('.')[0]
        # save_path = save_out + out_name + '.png'
        # cv2.imwrite(save_path,"trajectory_mask" )

