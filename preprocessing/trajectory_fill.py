import numpy as np
import cv2
import os
import ast

save_out = '../simple_dataset/trajectory_mask/'
tra_txt_path = os.path.expanduser('../runs/track/yolov5/runs/train/exp5/weights/best_osnet_x0_25_market1501/tracks/000050.txt')
img_path = '../simple_dataset/images/train/000050/000050_19.jpg'

tra_list = []
mask_list = []
tra_id1 = '1'
tra_id2 = '2'

with open(tra_txt_path, 'r', encoding='UTF-8') as f:
    for line in f:
        tra_dic = ast.literal_eval(line)
        for k, v in tra_dic.items():
            if k == tra_id1:
                tra_list.append(v)
            if k == tra_id2:
                tra_list.append(v)

imgs = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
tra_mask = np.zeros(imgs.shape[:2], dtype=np.uint8)
for i in tra_list:
    tra_mask = cv2.circle(tra_mask,(int(i[0]), int(i[1])), 5, (255, 255, 255), -1)
    mask_list.append(tra_mask)
    for k in mask_list:
        tra_image = cv2.add(imgs, np.zeros(imgs.shape[:2], dtype=np.uint8), mask=k)
        cv2.imwrite(r'C:\Users\YIHANG\PycharmProjects\HTAD\simple_dataset\trajectory_mask/000050_mask.jpg', k)

