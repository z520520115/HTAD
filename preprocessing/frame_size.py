from PIL import Image  # python3安装pillow库
import os.path
import glob


def convertSize(jpgfile, outdir, width=1024, height=436):  # 图片的大小256*256
    img = Image.open(jpgfile)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
        '''
        if new_img.mode == 'P':
            new_img = new_img.convert("RGB")
        if new_img.mode == 'RGBA':
            new_img = new_img.convert("RGB")
        '''
        new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)


for jpgfile in glob.glob(r"C:\Users\YIHANG\PycharmProjects\folwnet2\flownet2-pytorch\datasets\test/*.png"):  # 修改该文件夹下的jpg图片
    convertSize(jpgfile, r"C:\Users\YIHANG\PycharmProjects\folwnet2\flownet2-pytorch\datasets\test/")  # 另存为的文件夹路径
