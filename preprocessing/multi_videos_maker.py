import cv2
import os

video_frames_path = r'C:/Users/YIHANG/PycharmProjects/HTAD_dataset/video_frames/'

# 得到视频size
def get_size(img):
    size = (img.shape[1], img.shape[0])
    return size

# 得到该文件夹下帧的数量
def get_num(folder):
    count = 0
    for file in folder:
        ext = os.path.splitext(file)[-1].lower()
        if ext == '.jpg':
            count += 1
    return count

# 将一个文件夹下所有帧存入列表, 并得到size
def get_frames(path):
    frames_array = []
    for root, dirs, files in os.walk(path):
        files = str_to_int(files)
        files.sort()
        files = int_to_str(files)
        for f in files:
            img = cv2.imread(root + "/" + f)
            frames_array.append(img)
    return frames_array

# 列表元素str转int
def str_to_int(list):
    new_list = []
    for i in list:
        new_list.append(int(i[:-4]))
    list = new_list
    return list

# 列表元素str转int
def int_to_str(list):
    new_list = []
    for i in list:
        new_list.append(str(i) + ".jpg")
    list = new_list
    return list

# 制作视频
def get_videos(video_path, size, count, array):
    videos = cv2.VideoWriter(video_path, 0x7634706d, 25, size)
    for i in range(count):
        videos.write(array[i])
    videos.release()
    print("make " + video_path[52:62] + " successfully！")
    return videos


video_frames = []
frames_count = []

for root, dirs, files in os.walk(video_frames_path, topdown=True):
    for j in dirs:
        video_frames.append(root + j + "/")
    frames_count.append(get_num(files))

for idx, fra in enumerate(video_frames):
    frames_array = get_frames(fra)
    get_videos(fra[:45] + "videos" + fra[57:-1] + ".mp4", get_size(frames_array[0]), len(frames_array), frames_array)


