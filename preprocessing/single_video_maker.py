import cv2
import os

image = cv2.imread('C:/Users/YIHANG/PycharmProjects/HTAD_dataset/video_frames/000033/109.jpg')
size = (image.shape[1], image.shape[0])
videowrite = cv2.VideoWriter(r'C:/Users/YIHANG/PycharmProjects/HTAD_dataset/videos/000033.mp4' ,0x7634706d, 30, size)

def str_to_int(list):
    new_list = []
    for i in list:
        new_list.append(int(i[:-4]))
    list = new_list
    return list

def int_to_str(list):
    new_list = []
    for i in list:
        new_list.append(str(i) + ".jpg")
    list = new_list
    return list

# 读取所有img存入列表
img_array=[]
for root, dirs, files in os.walk(r'C:\Users\YIHANG\PycharmProjects\HTAD_dataset\video_frames\000033'):
    files = str_to_int(files)
    files.sort()
    files = int_to_str(files)
    for f in files:
        img = cv2.imread(root + "/" + f)
        if img is None:
            print(files + " is error!")
            continue
        img_array.append(img)

# 将读取的img存储为视频
for i in range(52):
    videowrite.write(img_array[i])
videowrite.release()
print('end!')