import os
import fileinput
import tifffile as tif
import cv2
import numpy as np
import glob
import os

def writefilename(file = 'test', str=''):
    """写入文件名"""
    # if os.path.exists(file):
    #     os.remove(file)
    f = open(file, 'w')
    for i in str:
        f.write(i)
    f.write('\n')
    f.close()

def readfilename(file = 'test'):
    """读入文件名"""
    list = []
    f = open(file, 'r')
    for lines in f:
        list.append(lines.replace("\n", ""))
    f.close()
    return list[0]

def cutBigImage(src = '', dst = '', size = 512, overlap = 256):
    """重叠裁剪大图"""
    # read image
    # ori_image = tif.imread(src)
    ori_image = cv2.imread(src)
    # ori_image = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    print(ori_image.shape)
    print('Start to clip ')

    # get iamge width and height
    ysize, xsize = ori_image.shape[0], ori_image.shape[1]
    dert = size - overlap
    h_n = (ysize - size) // dert
    if ((ysize - size)) % dert == 0:
        h_step = h_n
    else:
        h_step = h_n + 1

    w_n = (xsize - size) // dert
    if ((xsize - size) % dert) == 0:
        w_step = w_n
    else:
        w_step = w_n + 1

    # 沿X轴方向裁剪
    k = 0
    for h in range(h_step):
        for w in range(w_step):
            image_sample = ori_image[(h * dert):(h * dert + size), (w * dert):(w * dert + size), :]
            # tif.imsave(dst + str(k).zfill(4) + '.tif', image_sample)
            cv2.imwrite(dst + str(k).zfill(4) + '.tif', image_sample)
            k = k + 1
        # 顶着最右边裁剪
        a = ori_image[(h * dert):(h * dert + size), -size:, :]
        # tif.imsave(dst + str(k).zfill(4) + '.tif', a)
        cv2.imwrite(dst + str(k).zfill(4) + '.tif', a)
        k = k + 1
    for w in range(w_step):
        b = ori_image[-size:, (w * dert):(w * dert + size), :]
        # tif.imsave(dst + str(k).zfill(4) + '.tif'.format(k), b)
        cv2.imwrite(dst + str(k).zfill(4) + '.tif'.format(k), b)
        k = k + 1

    c = ori_image[-size:, -size:, :]
    # tif.imsave(dst + str(k).zfill(4) + '.tif', c)
    cv2.imwrite(dst + str(k).zfill(4) + '.tif', c)

def cutBigLabel(src = '', dst = '', size = 512, overlap = 256):
    """重叠裁剪大幅的单通标签"""
    # read image
    ori_image = tif.imread(src)
    print(ori_image.shape)
    print('Start to clip {}')

    # get iamge width and height
    ysize, xsize = ori_image.shape[0], ori_image.shape[1]

    dert = size - overlap
    h_n = (ysize - size) // dert
    if ((ysize - size)) % dert == 0:
        h_step = h_n
    else:
        h_step = h_n + 1

    w_n = (xsize - size) // dert
    if ((xsize - size) % dert) == 0:
        w_step = w_n
    else:
        w_step = w_n + 1

    # 沿X轴方向裁剪
    k = 0
    for h in range(h_step):
        for w in range(w_step):
            image_sample = ori_image[(h * dert):(h * dert + size), (w * dert):(w * dert + size)]
            tif.imsave(dst + str(k).zfill(4) + '.tif',
                       image_sample)
            k = k + 1
        # 顶着最右边裁剪
        a = ori_image[(h * dert):(h * dert + size), -size:]
        tif.imsave(dst + str(k).zfill(4) + '.tif', a)
        k = k + 1
    for w in range(w_step):
        b = ori_image[-size:, (w * dert):(w * dert + size)]
        tif.imsave(dst + str(k).zfill(4) + '.tif'.format(k), b)
        k = k + 1

    c = ori_image[-size:, -size:]
    tif.imsave(dst + str(k).zfill(4) + '.tif', c)

def mergeBigLabel(src = '', dst = '', ori = '', size = 512, overlap = 256):
    '''合并裁剪图片'''

    #读取原始影像信息
    # ori_image = tif.imread(ori)
    ori_image = cv2.imread(ori)
    ori_image = ori_image[..., 0:3]
    ysize, xsize = ori_image.shape[0], ori_image.shape[1]

    # 新建拼接影像
    tmp = np.zeros([ysize, xsize]).astype(np.uint8)

    predict_list = []
    paths = glob.glob(src)
    paths = sorted(paths)
    for i in paths:
        img = tif.imread(i)
        predict_list.append(img)

    dert = size - overlap
    h_n = (ysize - size) // dert
    if ((ysize - size)) % dert == 0:
        h_step = h_n
    else:
        h_step = h_n + 1

    w_n = (xsize - size) // dert
    if ((xsize - size) % dert) == 0:
        w_step = w_n
    else:
        w_step = w_n + 1

    # 沿X轴方向拼接
    k = 0
    for h in range(h_step):
        for w in range(w_step):
            tmp[(h * dert):(h * dert + size), (w * dert):(w * dert + size)] = predict_list[k]
            k = k +1
        # 拼接最右列的图片
        tmp[(h * dert):(h * dert + size), (xsize - size) : xsize] = predict_list[k]
        k = k + 1
    # 拼接最下一行影像，右下角的单独处理
    for w in range(w_step):
        tmp[-size:, (w * dert):(w * dert + size)] = predict_list[k]
        k = k + 1
    tmp[-size:, -size:] = predict_list[k]
    tif.imsave(dst, tmp)

def sort():
    paths = os.listdir('E:\\temp\\cut\\')
    # print(paths)

if __name__ == '__main__':
    # srcdir = '/media/dell/DATA/cl/pytorch/Khaos/dataset/origin/Subset1.tif'
    # dstdir = '/media/dell/DATA/cl/pytorch/Khaos/dataset/test/Subset1_'
    # cutBigImage(src = srcdir, dst = dstdir)

    srcdir = '/media/dell/DATA/cl/pytorch/Khaos/result/DeepLabV3Plus/*.tif'
    dstdir = '/media/dell/DATA/cl/pytorch/Khaos/dataset/origin/Subset1_lab.tif'
    oridir = '/media/dell/DATA/cl/pytorch/Khaos/dataset/origin/Subset1.tif'
    mergeBigLabel(src=srcdir, dst=dstdir, ori=oridir)








































