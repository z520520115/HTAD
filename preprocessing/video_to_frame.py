import cv2

vidcap = cv2.VideoCapture(r'C:\Users\YIHANG\Desktop\crash-1500/000001.mp4')
success, image = vidcap.read()
count = 0
success = True
while success:
    success, image = vidcap.read()
    cv2.imwrite(r"C:\Users\YIHANG\PycharmProjects\folwnet2\flownet2-pytorch\datasets\test/frame%d.png" % count,image)
    if cv2.waitKey(10) == 27:
        break
    count+=1