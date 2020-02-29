import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import testdata_loader
from test import Solvertest
import data_processing
import random

def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net','DA_Net','DD_Net','DeepLabV3Plus','FCN']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net/DA_Net/DD_Net/DeepLabV3Plus/FCN')
        print('Your input for model_type was %s%config.model_type')
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)


    print(config)
    train_loader = []
    valid_loader = []
    test_loader  = []
    if config.mode == 'train':
        train_loader = get_loader(image_path=config.train_path,
                                  image_size=config.image_size,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  mode='train',
                                  augmentation_prob=config.augmentation_prob)
        valid_loader = get_loader(image_path=config.valid_path,
                                  image_size=config.image_size,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  mode='valid',
                                  augmentation_prob=0.)
    if config.mode == 'test':
        # test_loader = get_loader(image_path=config.test_path,
        #                          image_size=config.image_size,
        #                          batch_size=config.batch_size,
        #                          num_workers=config.num_workers,
        #                          mode='test',
        #                          augmentation_prob=0.)

        # 清理文件夹
        for i in os.listdir(config.test_path):
            file_data = config.test_path + '/' + i
            if os.path.isfile(file_data) == True:
                os.remove(file_data)
        for i in os.listdir(config.result_path):
            file_data = config.result_path + '/' + i
            if os.path.isfile(file_data) == True:
                os.remove(file_data)

        data_processing.cutBigImage(config.origin_path, config.test_path + 'img_',config.image_size,int(config.image_size/2))
        test_loader = testdata_loader.get_loader(image_path=config.test_path,
                                 image_size=config.image_size,
                                 batch_size=config.batch_size,
                                 num_workers=config.num_workers,
                                 mode='test',
                                 augmentation_prob=0.)

    solver = Solver(config, train_loader, valid_loader, test_loader)
    solvertest = Solvertest(config, train_loader, valid_loader, test_loader)

    
    # Train and sample the images 训练和采样图像
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solvertest.test()
        data_processing.mergeBigLabel(src=config.result_path + '/*.tif', dst=config.merge_path, ori=config.origin_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--num_epochs_decay', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--model_type', type=str, default='DeepLabV3Plus',
                        help='U_Net/R2U_Net/AttU_Net/R2AttU_Net/DA_Net/DD_Net/DeepLabV3Plus/FCN')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--train_path', type=str, default='./dataset/train/')
    parser.add_argument('--valid_path', type=str, default='./dataset/valid/')
    parser.add_argument('--test_path', type=str, default='./dataset/test/')
    parser.add_argument('--result_path', type=str, default='./result/')

    parser.add_argument('--origin_path', type=str, default='./dataset/origin/Subset1.tif')
    parser.add_argument('--merge_path', type=str, default='./dataset/origin/Subset1_lab.tif')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)
