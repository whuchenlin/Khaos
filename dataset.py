import os
import argparse
import random
import shutil
from shutil import copyfile
from misc import printProgressBar


def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s'%dir_path)
    os.makedirs(dir_path)
    print('Create path - %s'%dir_path)

def main(config):

    rm_mkdir(config.train_path)
    rm_mkdir(config.train_GT_path)
    rm_mkdir(config.valid_path)
    rm_mkdir(config.valid_GT_path)
    rm_mkdir(config.test_path)
    rm_mkdir(config.test_GT_path)

    filenames = os.listdir(config.origin_data_path)
    data_list = []
    GT_list = []

    for filename in filenames:
        # 分离文件名和扩展名
        #ext = os.path.splitext(filename)[-1]
        #if ext =='.jpg':
            # filename = filename.split('_')[-1][:-len('.jpg')]
            # # 在训练样本图片加上前缀ISIC_，GT_list定义的名称用来干啥？
        # data_list.append('ISIC_'+filename+'.jpg')
        # GT_list.append('ISIC_'+filename+'_segmentation.png')
        data_list.append(filename)
        GT_list.append(filename)

    # 分配测试集 验证集 训练集（数量）
    num_total = len(data_list)
    num_train = int((config.train_ratio/(config.train_ratio+config.valid_ratio+config.test_ratio))*num_total)
    num_valid = int((config.valid_ratio/(config.train_ratio+config.valid_ratio+config.test_ratio))*num_total)
    num_test = num_total - num_train - num_valid

    print('\nNum of train set : ',num_train)
    print('\nNum of valid set : ',num_valid)
    print('\nNum of test set : ',num_test)

    # 将元素打乱
    Arange = list(range(num_total))
    random.shuffle(Arange)

    # 训练集
    for i in range(num_train):

        # 删除一个返回一个
        idx = Arange.pop()

        # 把原始下载的训练影像改名 COPY到新的 dataset/train中去
        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.train_path,data_list[idx])
        copyfile(src, dst)

        #对应的GT图，同样的COPY操作
        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.train_GT_path, GT_list[idx])
        copyfile(src, dst)

        printProgressBar(i + 1, num_train, prefix = 'Producing train set:', suffix = 'Complete', length = 50)
        
    # 验证集
    for i in range(num_valid):
        idx = Arange.pop()

        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.valid_path,data_list[idx])
        copyfile(src, dst)
        
        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.valid_GT_path, GT_list[idx])
        copyfile(src, dst)

        printProgressBar(i + 1, num_valid, prefix = 'Producing valid set:', suffix = 'Complete', length = 50)

    # 测试集
    for i in range(num_test):
        idx = Arange.pop()

        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.test_path,data_list[idx])
        copyfile(src, dst)
        
        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.test_GT_path, GT_list[idx])
        copyfile(src, dst)


        printProgressBar(i + 1, num_test, prefix = 'Producing test set:', suffix = 'Complete', length = 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--valid_ratio', type=float, default=0.05)
    parser.add_argument('--test_ratio', type=float, default=0.05)

    # data path
    parser.add_argument('--origin_data_path', type=str, default='./dataset/db')
    parser.add_argument('--origin_GT_path', type=str, default='./dataset/lab')
    
    parser.add_argument('--train_path', type=str, default='./dataset/train/')
    parser.add_argument('--train_GT_path', type=str, default='./dataset/train_GT/')
    parser.add_argument('--valid_path', type=str, default='./dataset/valid/')
    parser.add_argument('--valid_GT_path', type=str, default='./dataset/valid_GT/')
    parser.add_argument('--test_path', type=str, default='./dataset/test/')
    parser.add_argument('--test_GT_path', type=str, default='./dataset/test_GT/')

    config = parser.parse_args()
    print(config)
    main(config)