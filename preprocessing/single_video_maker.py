import cv2
import os

'''
多帧转单个视频
'''

image = cv2.imread(r'C:\Users\YIHANG\PycharmProjects\AutomaticScenarioGenerationSystem\output\4D\output_frame\0002/0002_00000_4D.png')
size = (image.shape[1], image.shape[0])
videowrite = cv2.VideoWriter(r'C:\Users\YIHANG\PycharmProjects\AutomaticScenarioGenerationSystem\output\4D\output_frame/0002.mp4' ,0x7634706d, 25, size)

def str_to_int(list):
    new_list = []
    for i in list:
        new_list.append(int(i[8:10]))
    list = new_list
    return list

def int_to_str(list):
    new_list = []
    for i in list:
        new_list.append(str(i) + ".png")
    list = new_list
    return list

# 读取所有img存入列表
img_array=[]
for root, dirs, files in os.walk(r'C:\Users\YIHANG\PycharmProjects\AutomaticScenarioGenerationSystem\output\4D\output_frame\0002'):
    # files = str_to_int(files)
    # files.sort()
    # files = int_to_str(files)
    for f in files:
        img = cv2.imread(root + "/" + f)
        if img is None:
            print(files + " is error!")
            continue
        img_array.append(img)

# 将读取的img存储为视频
for i in range(100):
    videowrite.write(img_array[i])
videowrite.release()
print('end!')