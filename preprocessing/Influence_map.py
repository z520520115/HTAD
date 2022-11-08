import os
import cv2
import numpy as np
import math

PATH = os.path.dirname(os.path.realpath(__file__))


def get_video_frame(video_name):
    video_path = PATH + '/vid_yolov5_deepsort/{}.mp4'.format(video_name)
    video = cv2.VideoCapture(video_path)

    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(height), int(width), 3)

    return size


def draw_accident_circle(img, COLOR, default_size, ac_idx1=None, ac_idx2=None):
    # 这是一个控制圆的大小和亮度的变量。 你可以改变这个
    # This is a variable that controls the size and brightness of the circle.

    # Please comment to manually control (要手动控制，请加注)
    # CIRCLE_RATE = 100 # Default
    width, height, _ = img.shape
    CIRCLE_RATE = int(width * height / 10000)

    if None not in [ac_idx1, ac_idx2]:
        ac_idx1 = [int(ac_idx1[0]), int(ac_idx1[1])]
        ac_idx2 = [int(ac_idx2[0]), int(ac_idx2[1])]

        # length = math.dist(ac_idx1, ac_idx2)
        length = math.sqrt( ((int(ac_idx1[0])-int(ac_idx2[0]))**2)+((int(ac_idx1[1])-int(ac_idx2[1]))**2) )
        size = default_size + 1 + int(10 * (CIRCLE_RATE / (length + 1)))
        if size > 15:
            size = 15

        copied_img = np.zeros(img.shape, np.uint8)
        cv2.circle(copied_img, ac_idx1, size, [0, 0, COLOR], -1)
        img = cv2.add(copied_img, img)

        copied_img = np.zeros(img.shape, np.uint8)
        cv2.circle(copied_img, ac_idx2, size, [0, 0, COLOR], -1)
        img = cv2.add(img, copied_img)
    else:
        size = default_size + 1
        if ac_idx1 is not None:
            copied_img = np.zeros(img.shape, np.uint8)
            cv2.circle(copied_img, ac_idx1, size, [0, 0, COLOR], -1)
            img = cv2.add(copied_img, img)

        elif ac_idx2 is not None:
            copied_img = np.zeros(img.shape, np.uint8)
            cv2.circle(copied_img, ac_idx2, size, [0, 0, COLOR], -1)
            img = cv2.add(img, copied_img)

    return img


def make_image(txt_name, video_name, accident_idx1, accident_idx2):
    file = PATH + '/vid_yolov5_deepsort/{}.txt'.format(txt_name)
    size = get_video_frame(video_name)

    frame = {}
    with open(file) as f:
        while True:
            line = f.readline()
            if not line:
                frame_time = frame_idx
                break
            line = eval(line[0:-1])
            character = list(line.keys())[1]
            frame_idx, location = list(line.values())
            location = [int(location[0]), int(location[1])]

            if frame_idx not in frame:
                frame[frame_idx] = []
            frame[frame_idx].append([character, location])

    no_accident_img = np.zeros(size, np.uint8)
    accident_img = np.zeros(size, np.uint8)

    ColorToTime = 250 / int(frame_time)
    default_size = int(((size[0] / 720) + (size[1] / 1280)) / 2 * 5)
    if default_size <= 0:
        default_size = 2

    accident = {
        accident_idx1: None,
        accident_idx2: None
    }
    for step in range(frame_time + 1):
        if step not in frame:
            continue
        for data in frame[step]:
            if data[0] in [accident_idx1, accident_idx2]:
                accident[data[0]] = data[1]
                continue

            new_img = np.zeros(size, np.uint8)
            cv2.circle(new_img, data[1], default_size + 1, [5 + ColorToTime * step, 0, 0], -1)
            no_accident_img = cv2.add(no_accident_img, new_img)

        accident_img = draw_accident_circle(accident_img, 5 + ColorToTime * step, default_size, accident[accident_idx1], accident[accident_idx2])

        complete_img = cv2.add(accident_img, no_accident_img)

        cv2.imwrite(PATH+"/influence_map/{}/{}_{}.png".format(txt_name, txt_name, step), complete_img)

if __name__ == '__main__':
    NAME = '001500'
    ac1 = '1'
    ac2 = '7'
    make_image(NAME, NAME, ac1, ac2)
