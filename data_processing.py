import os
import fileinput

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

if __name__ == '__main__':
    # writefilename()
    # readfilename()
    print(readfilename(file = "path_image"))








































