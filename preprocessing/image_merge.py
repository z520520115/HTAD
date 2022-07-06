import cv2
import os

# video_path = r'C:/Users/YIHANG/PycharmProjects/HTAD/dataset/video_frames/'
#
# # 计算共有多少图片
# count = 0
# video_frames = []
# for root, dirs, files in os.walk(video_path, topdown=True):
#     for i in dirs:
#         video_frames.append(root + i + "/")
#
#     for file in files:
#         ext = os.path.splitext(file)[-1].lower()
#         if ext == '.jps':
#             count += 1
#             print(count)

image = cv2.imread('C:/Users/YIHANG/PycharmProjects/HTAD/dataset/video_frames/000049/19.jpg')
size = (image.shape[1], image.shape[0])
videowrite = cv2.VideoWriter(r'C:/Users/YIHANG/PycharmProjects/HTAD/dataset/videos/000049.mp4' ,0x7634706d, 20, size)

# 读取所有img存入列表
img_array=[]
for root, dirs, files in os.walk(r'C:\Users\YIHANG\PycharmProjects\HTAD\dataset\video_frames\000049'):
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
