import os
import shutil

'''
视频frame数据整理, 在每个文件夹中生成两个新文件夹 accident, no_accident 移动frame进这两个文件
'''

path = 'C:/Users/YIHANG/Desktop/CADP/test'
# 遍历全frames路径
for root, dirs, files in os.walk(path):
    for dir_idx, j in enumerate(dirs):

        dir_path = os.path.join(path, j)
        acc_path = os.path.join(dir_path, 'accident')
        no_acc_path = os.path.join(dir_path, 'no_accident')

        files_list = os.listdir(dir_path)
        files_list.sort(key=lambda x:int(x[7:-4]))

        if os.path.exists(acc_path) == False:
            os.mkdir(acc_path)
        if os.path.exists(no_acc_path) == False:
            os.mkdir(no_acc_path)

        for idx, frame in enumerate(files_list):
            frame_path = os.path.join(dir_path, frame)
            if idx < 25:
                shutil.move(frame_path, no_acc_path)
            else:
                shutil.move(frame_path, acc_path)

