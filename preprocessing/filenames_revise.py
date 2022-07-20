import os

# 改名字前记得备份!!!
path = (r'C:\Users\YIHANG\PycharmProjects\HTAD_dataset\video_frames')

for root, dirs, files in os.walk(path):
    for i in dirs:
        f_path = (r'C:\Users\YIHANG\PycharmProjects\HTAD_dataset\video_frames' + '/' + i)
        for r, d, f in os.walk(f_path):
            for idx, j in enumerate(f):
                oldname = f_path + '/' + j
                newname = f_path + '/' + i + '_' + str(idx) + '.jpg'
                os.rename(oldname, newname)
