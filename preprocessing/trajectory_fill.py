import numpy as np
import cv2
import os

file_root = '../simple_dataset/images/train/000003/'
file_list = os.listdir(file_root)
save_out = '../simple_dataset/trajectory_mask/'
tra_txt_path = os.path.expanduser('../runs/track/yolov5/runs/train/exp5/weights/best_osnet_x0_25_market1501/tracks/000003.txt')

tra_l1 = []
tra_l2 = []

with open(tra_txt_path, 'r', encoding='UTF-8') as f:
    tra_list = f.readlines()
tra_list = [c.strip() for c in tra_list]
# tra_list =

print(tra_list[80][1:3])
for i in range(len(tra_list)):
    if tra_list[i][1:3] is '2':
        tra_l1.append(tra_list[i][4:])
    elif tra_list[i][1:3].find('5'):
        tra_l2.append(tra_list[i][4:])
print(tra_l1)
print(tra_l2)

for img_name in file_list:
    if img_name.endswith('.jpg'):
        img_path = file_root + img_name
        imgs = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        tra_mask = np.zeros(imgs.shape[:2], dtype=np.uint8)

        tra_mask = cv2.circle(tra_mask, (299, 164), 10, (255, 255, 255), -1)
        tra_mask2 = cv2.circle(tra_mask, (304, 164), 10, (255, 255, 255), -1)
        tra_mask3 = cv2.circle(tra_mask, (312, 163), 10, (255, 255, 255), -1)
        combined_mask = tra_mask | tra_mask2 | tra_mask3
        tra_image = cv2.add(imgs, np.zeros(imgs.shape[:2], dtype=np.uint8), mask=combined_mask)
        cv2.imwrite(r'C:\Users\YIHANG\PycharmProjects\HTAD\simple_dataset\trajectory_mask/mask1.jpg', combined_mask)
        # out_name = img_name.split('.')[0]
        # save_path = save_out + out_name + '.png'
        # cv2.imwrite(save_path,"trajectory_mask" )
