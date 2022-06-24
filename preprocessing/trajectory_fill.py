import numpy as np
import cv2
import os
import ast
import glob

# save_out = '../simple_dataset/trajectory_mask_0624/'
tra_txt_path = os.path.expanduser('../runs/track/yolov5/runs/train/exp5/weights/best_osnet_x0_25_market1501/tracks/000003.txt')
img_path = '../simple_dataset/images/train/000003/000003_19.jpg'

image_path_list = []
image_path_list.extend(glob.glob(r"../simple_dataset/images/train/000050/*.jpg"))

tra_list = []
mask_list = []
fream_idx = 2
tra_id1 = '1'
tra_id2 = '5'


def get_n_channel(img):
	if img.ndim == 2:
		print('通道数：1')
		return 1
	else:
		print('图像包含多个通道')
		return img.shape[2]


# 打开轨迹txt文件将选定帧坐标存入列表
with open(tra_txt_path, 'r', encoding='UTF-8') as f:
    for line in f:
        tra_dic = ast.literal_eval(line)
        for k, v in tra_dic.items():
            if k == tra_id1:
                tra_list.append(v)
            if k == tra_id2:
                tra_list.append(v)

# 创建轨迹mask, 一个坐标为一个半径为5的mask
imgs = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
tra_mask = np.zeros(imgs.shape[:2], dtype=np.uint8)

# 创建单张轨迹mask, 将所有点mask叠加
for i in tra_list:
    tra_mask = cv2.circle(tra_mask,(int(i[0]), int(i[1])), 5, (255, 255, 255), -1)
    mask_list.append(tra_mask)
    for k in mask_list:
        # tra_image = cv2.add(imgs, np.zeros(imgs.shape[:2], dtype=np.uint8), mask=k)
        tra_image_gray = cv2.imwrite(r'C:\Users\YIHANG\PycharmProjects\HTAD\simple_dataset\trajectory_mask_0624/000003_mask.jpg', k)
        tra_image_color = np.concatenate((tra_image_gray, tra_image_gray, tra_image_gray), axis=-1)
        cv2.imwrite(r'C:\Users\YIHANG\PycharmProjects\HTAD\simple_dataset\trajectory_mask_0624/000003_mask.jpg', tra_image_color)
        get_n_channel(tra_image_color)
# # 创建多张所有坐标的mask
# for i, val in enumerate(tra_list):
#     imgs = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     # tra_image = cv2.add(imgs, np.zeros(imgs.shape[:2], dtype=np.uint8), mask=tra_mask)
#     if i % 2 == 0: # 记录第一条轨迹, 用i序列筛选第一条
#         tra_mask1 = np.zeros(imgs.shape[:2], dtype=np.uint8)
#         tra_mask1 = cv2.circle(tra_mask1, (int(val[0]), int(val[1])), 5, (255, 255, 255), -1)
#     elif i % 2 == 1 : # 记录第二条轨迹, 用i序列筛选第一条
#         tra_mask2 = np.zeros(imgs.shape[:2], dtype=np.uint8)
#         tra_mask2 = cv2.circle(tra_mask2, (int(val[0]), int(val[1])), 5, (255, 255, 255), -1)
#         cv2.imwrite(r'C:\Users\YIHANG\PycharmProjects\HTAD\simple_dataset\trajectory_mask_0615\train\000050_' + str(fream_idx) + '.png', tra_mask1 + tra_mask2)
#
#         fream_idx += 1