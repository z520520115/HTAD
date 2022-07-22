import os
from shutil import copy

'''
补充丢失的轨迹帧, 通过判断当前帧和下一帧的差值, 补充丢失的帧, 例如2.jpg, 5.jpg时 插入3.jpg, 4.jpg
'''

path = r'C:\Users\YIHANG\PycharmProjects\HTAD_dataset\trajectory_mask/'

# 列表元素str转int
def str_to_int(list):
    new_list = []
    for i in list:
        new_list.append(int(i[7:-4]))
    list = new_list
    return list

# 列表元素str转int
def int_to_str(list, file_name):
    new_list = []
    for i in list:
        new_list.append(file_name + '_' + str(i) + ".jpg")
    list = new_list
    return list

for root, dirs, files in os.walk(path):

    # 列表排序先
    f = str_to_int(files)
    f.sort()
    f = int_to_str(f, files[0][:6])

    # 实际第一帧的path
    path_first = path + f[0]
    # 实际最后一帧的path
    path_finally = path + f[-1]
    # 第2帧的path
    path_2frame = path + f[0][:7] + "2.jpg"
    # 第51帧的path
    path_51frame = path + f[0][:7] + "51.jpg"

    # 如果第二帧不存在就复制第一项给第二帧
    if os.path.exists(path_2frame) is False:
        copy(path_first, path_2frame)
        f.insert(0, f[0][:7] + "2.jpg")

    # 如果第51帧不存在就复制最后一项给第51帧
    if os.path.exists(path_51frame) is False:
        copy(path_finally, path_51frame)
        f.append(f[0][:7] + "51.jpg")

    for idx, i in enumerate(f):
        # 下一帧的path
        path_second = path + str(f[idx + 1])
        # 复制为当前帧的path
        path_copy = path + i[:7] + str(int(f[idx + 1][7:-4]) - 1) + ".jpg"
        # count为当前帧减下一帧的值
        count = int(f[idx][7:-4]) - int(f[idx+1][7:-4])

        # 如果count等于-2, 就复制下一帧为当前帧
        if count == -2:
            copy(path_second, path_copy)
            print("copy " + i[:7] + str(int(f[idx+1][7:-4])-1) + " 成功")

        # 如果count小于-1, 就复制(n/-1)-1份
        elif count < -1:
            n = int(count/-1)
            for j in range(1, n):
                path_second = path + str(f[idx + 1])
                path_copy = path + i[:7] + str(int(f[idx + 1][7:-4]) - j) + ".jpg"
                copy(path_second, path_copy)
