import os

def get_num(folder):
    count = 0
    for file in folder:
        ext = os.path.splitext(file)[-1].lower()
        if ext == '.jpg':
            count += 1
    return count

path = (r'C:\Users\YIHANG\PycharmProjects\HTAD_dataset\video_frames/')

for root, dirs, files in os.walk(path):

    for idx, j in enumerate(dirs):
        path_f = path + dirs[idx]
        count = 0
        for r, d, f in os.walk(path_f):
            for k in f:
                ext = os.path.splitext(k)[-1].lower()
                if ext == '.jpg':
                    count += 1
        print(j + "=" + str(count))