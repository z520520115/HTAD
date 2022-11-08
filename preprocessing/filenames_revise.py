import os

'''
批量修改当前帧idx, 例如将64.jpg, 65.jpg, 66.jpg改为000003_1.jpg, 000003_2.jpg, 000003_3.jpg 或者png
'''

# 根据_来找数字, 3位数, 2位数, 1位数
def str_to_int(list):
    new_list = []
    for i in list:
        new_list.append(int(i[:i.index('.')]))
    list = new_list
    return list

# 列表元素str转int (因为没备份名字都乱掉了, 写了一个混乱的函数)
def int_to_str(list, file_name):
    new_list = []
    new_list.append(str(list[0]) + ".jpg")
    for i in list[1:]:
        new_list.append(str(i) + ".jpg")
    list = new_list
    return list

ori_name_list = []
# 改名字前记得备份!!!
path = (r'C:\Users\YIHANG\Desktop\001371')

for root, dirs, files in os.walk(path):
    for i in dirs:
        f_path = (path + '/' + i)
        for r, d, f in os.walk(f_path):

            # 字符串转整形→排序→整形转字符串
            f = str_to_int(f)
            f.sort()
            f = int_to_str(f, i)

            # 改成新名字
            for idx, j in enumerate(f):

                oldname = f_path + '/' + j
                newname = f_path + '/' + i + '_' + str(idx) + '.jpg'
                os.rename(oldname, newname)

