import numpy as np
import cv2
import os
import ast

tra_txt_path = os.path.expanduser('../runs/track9/000049.txt')
img_path = '../dataset/video_frames/000049/65.jpg'

tra_list = []
frame_idx_l = []
frame_idx = 1
tra_id1 = '20'
tra_id2 = '22'

def get_n_channel(img):
	if img.ndim == 2:
		print('通道数：1')
		return 1
	else:
		print('图像包含多个通道')
		return img.shape[2]

def channel_1to3(mask):
    im = mask[:, :, np.newaxis]
    tra_image_color = im.repeat([3], axis=2)
    return tra_image_color

# 打开轨迹txt文件将选定帧坐标存入列表
with open(tra_txt_path, 'r', encoding='UTF-8') as f:
    for line in f:
        tra_dic = ast.literal_eval(line) # 字符串转换成字典
        for k, v in tra_dic.items():
            if k == tra_id1:
                tra_list.append(v)
                frame_idx_l.append(tra_dic['frame_idx'])
            if k == tra_id2:
                tra_list.append(v)
                frame_idx_l.append(tra_dic['frame_idx'])
'''
# 创建轨迹mask, 一个坐标为一个半径为5的mask
imgs = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
tra_mask = np.zeros(imgs.shape[:2], dtype=np.uint8)

# 创建单张轨迹mask, 将所有点mask叠加
for i in tra_list:
    tra_mask = cv2.circle(tra_mask,(int(i[0]), int(i[1])), 5, (255, 255, 255), -1)
    mask_list.append(tra_mask)
    for k in mask_list:
        cv2.imwrite(r'C:/Users\YIHANG\PycharmProjects\HTAD\dataset/trajectory_mask/000003.jpg', channel_1to3(k))
get_n_channel(channel_1to3(k))

# 创建多张所有坐标的mask
for i, val in enumerate(tra_list):
    # imgs = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # tra_image = cv2.add(imgs, np.zeros(imgs.shape[:2], dtype=np.uint8), mask=tra_mask)
    if i % 2 == 0: # 记录第一条轨迹, 用i序列筛选第一条
        tra_mask1 = np.zeros(imgs.shape[:2], dtype=np.uint8)
        tra_mask1 = cv2.circle(tra_mask1, (int(val[0]), int(val[1])), 5, (255, 255, 255), -1)
    elif i % 2 == 1 : # 记录第二条轨迹, 用i序列筛选第一条
        tra_mask2 = np.zeros(imgs.shape[:2], dtype=np.uint8)
        tra_mask2 = cv2.circle(tra_mask2, (int(val[0]), int(val[1])), 5, (255, 255, 255), -1)
        cv2.imwrite(r'C:/Users\YIHANG\PycharmProjects\HTAD\dataset/trajectory_mask/000003_' + str(fream_idx) + '.jpg', channel_1to3(tra_mask1) + channel_1to3(tra_mask2))
        fream_idx += 1
'''

################ 创建连续帧轨迹 ################
imgs = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
tra_mask = np.zeros(imgs.shape[:2], dtype=np.uint8)
for i, val in enumerate(tra_list):
    if i % 2 == 0: # 记录第一条轨迹, 用i序列筛选第一条
        tra_mask1 = cv2.circle(tra_mask, (int(val[0]), int(val[1])), 5, (255, 255, 255), -1)
    elif i % 2 == 1 : # 记录第二条轨迹, 用i序列筛选第一条
        tra_mask2 = cv2.circle(tra_mask, (int(val[0]), int(val[1])), 5, (255, 255, 255), -1)
        cv2.imwrite(r'C:\Users\YIHANG\PycharmProjects\HTAD\dataset\trajectory_mask\000049_' + str(frame_idx_l[i]) + '.jpg', channel_1to3(tra_mask1) + channel_1to3(tra_mask2))
