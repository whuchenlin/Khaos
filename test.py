import os
import numpy as np
from torch import optim
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
from module.models import danet
from deeplabv3plus import deeplab
import tifffile as tif


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Solvertest(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# 数据加载
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# 模型
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = torch.nn.BCELoss() # 损失函数
		self.augmentation_prob = config.augmentation_prob # 分割率

		# 超参
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# 训练设置
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# 路径设置
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		# 其他
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if   self.model_type =='U_Net':
			 self.unet = torch.nn.DataParallel(U_Net(img_ch=3,output_ch=self.output_ch),device_ids=[0])
		elif self.model_type =='R2U_Net':
			 self.unet = R2U_Net(img_ch=3,output_ch=self.output_ch,t=self.t)
		elif self.model_type =='AttU_Net':
			 self.unet = AttU_Net(img_ch=3,output_ch=self.output_ch)
		elif self.model_type == 'R2AttU_Net':
			 self.unet = torch.nn.DataParallel(R2AttU_Net(img_ch=3,output_ch=self.output_ch,t=self.t),device_ids=[0,1,2])
		elif self.model_type == 'DA_Net':
			 self.unet = torch.nn.DataParallel(danet.DANet(nclass=1, backbone='resnet101'),
											   device_ids=[0, 1, 2])
		elif self.model_type == 'DD_Net':
			 # self.unet = danet.DDNet(nclass=1, backbone='resnet50')
			 self.unet = torch.nn.DataParallel(danet.DDNet(nclass=1, backbone='resnet101'),
											   device_ids=[0,1])
		elif self.model_type == 'DeepLabV3Plus':
			 self.unet = torch.nn.DataParallel(deeplab.DeepLab(num_classes=1, backbone='resnet'),
											  device_ids=[0])

		self.optimizer = optim.Adam(list(self.unet.parameters()), self.lr, [self.beta1, self.beta2])
		self.unet.to(self.device)




	def test(self):

		# 模型-训练代数-学习率-下降代数-增强率
		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' % (
		self.model_type, self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))

		# U-Net Test
		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
		else:
			print('%s is no found，the path is error:%s' % (self.model_type, unet_path))
			return

		self.unet.train(False)
		self.unet.eval()

		for i, (images, filenames) in enumerate(self.test_loader):
			images = images.to(self.device)
			SR = F.sigmoid(self.unet(images))

			# 预测
			pre = SR
			pre[SR >= 0.5] = 255
			pre[SR < 0.5] = 0
			pre_ori = pre.data

			for j in range(pre_ori.shape[0]):
				# GPU传入CPU
				pre = pre_ori.cpu().numpy()[j]
				pre = np.squeeze(pre).astype(np.uint8)
				tif.imwrite(self.result_path + "/" + filenames[j], pre)
				print(filenames[j])
