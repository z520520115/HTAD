import os

def str_to_int(list):
    new_list = []
    for i in list:
        if i[-8] == "_":
            new_list.append(int(i[-7:-4]))
        elif i[-7] == "_":
            new_list.append(int(i[-6:-4]))
        elif i[-6] == "_":
            new_list.append(int(i[-5]))
    list = new_list
    return list

# 列表元素str转int
def int_to_str(list, file_name):
    new_list = []
    new_list.append(file_name + '_'
                    + file_name + '_'
                    # + file_name + '_'
                    + file_name + '_' +
                    str(list[0]) + ".jpg")
    for i in list[1:]:
        new_list.append(file_name + '_' +
                        file_name + '_' +
                        # file_name + '_' +
                        str(i) + ".jpg")
    list = new_list
    return list

ori_name_list = []
# 改名字前记得备份!!!
path = (r'C:\Users\YIHANG\PycharmProjects\HTAD_dataset\video_frames')

for root, dirs, files in os.walk(path):
    for i in dirs:
        f_path = (r'C:\Users\YIHANG\PycharmProjects\HTAD_dataset\video_frames' + '/' + i)
        for r, d, f in os.walk(f_path):

            # 字符串转整形→排序→整形转字符串
            f = str_to_int(f)
            f.sort()
            f = int_to_str(f, i)

            for idx, j in enumerate(f):

                oldname = f_path + '/' + j
                newname = f_path + '/' + i + '_' + str(idx) + '.jpg'
                os.rename(oldname, newname)
