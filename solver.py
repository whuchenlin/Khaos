import os
import numpy as np
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
from module.models import danet
from deeplabv3plus import deeplab
from torchsummary import summary
import visualization as visual
import csv
import tifffile as tif
import cv2

class Solver(object):
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
			 self.unet = R2AttU_Net(img_ch=3,output_ch=self.output_ch,t=self.t)
		elif self.model_type == 'DA_Net':
			 self.unet = danet.DANet(nclass=1, backbone='resnet50')
		elif self.model_type == 'DD_Net':
			 self.unet = torch.nn.DataParallel(danet.DDNet(nclass=1, backbone='resnet101'),device_ids=[0,1])
		elif self.model_type == 'DeepLabV3Plus':
			 self.unet = torch.nn.DataParallel(deeplab.DeepLab(num_classes=1, backbone='resnet'),device_ids=[0])

		self.optimizer = optim.Adam(list(self.unet.parameters()), self.lr, [self.beta1, self.beta2])
		self.unet.to(self.device)


	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		"""更新学习率"""
		lr = d_lr * (0.1 ** (g_lr // 30))
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

	def reset_grad(self):
		"""梯度下降缓存清理"""
		self.unet.zero_grad()



	def train(self):

		lr = self.lr
		best_iou = 0
		vis = visual.Visualization()
		vis.create_summary(self.model_type)

		'''训练'''
		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' % (self.model_type,
									self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))
		if os.path.isfile(unet_path):
			print('%s was existed' % (self.model_type))
			# 加载预训练模型
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
		else:
			print('%s was not created' % (self.model_type))

		for epoch in range(self.num_epochs):

			self.unet.train(True)
			epoch_loss = 0

			acc = 0.  # Accuracy
			SE = 0.  # Sensitivity
			SP = 0.  # Specificity
			PC = 0.  # Precision
			F1 = 0.  # F1 Score
			JS = 0.  # Jaccard Similarity
			DC = 0.  # Dice Coefficient
			length = 0

			for i, (images, GT) in enumerate(self.train_loader):

				images = images.to(self.device)
				GT = GT.to(self.device)

				# SR : Segmentation Result
				SR = self.unet(images)
				SR_probs = SR_eva= F.sigmoid(SR)
				SR_flat = SR_probs.view(SR_probs.size(0), -1)

				GT_flat = GT.view(GT.size(0), -1)
				loss = self.criterion(SR_flat, GT_flat)
				epoch_loss += loss.item()

				# Backprop + optimize
				self.reset_grad()
				loss.backward()
				self.optimizer.step()

				# 一个batch只做一次精度评定
				# 所以用影像总数来做平均是错的
				#  length += images.size(0)
				acc += get_accuracy(SR_eva, GT)
				SE += get_sensitivity(SR_eva, GT)
				SP += get_specificity(SR_eva, GT)
				PC += get_precision(SR_eva, GT)
				F1 += get_F1(SR_eva, GT)
				JS += get_JS(SR_eva, GT)
				DC += get_DC(SR_eva, GT)
				length += 1

				# Print the log info
				print('Batch [%d], Loss: %.4f' % (i, epoch_loss/(i+1)))

			acc = acc / length
			SE = SE / length
			SP = SP / length
			PC = PC / length
			F1 = F1 / length
			JS = JS / length
			DC = DC / length

			# Visualization
			vis.add_scalar(epoch, JS, 'IOU')
			vis.add_scalar(epoch, acc, 'acc')
			vis.add_scalar(epoch, SE, 'SE')
			vis.add_scalar(epoch, SP, 'SP')
			vis.add_scalar(epoch, PC, 'PC')
			vis.add_scalar(epoch, F1, 'F1')

			# Print the log info
			print(
				'Epoch [%d/%d], Loss: %.4f, \n[Training] acc: %.4f, PC: %.4f, SP: %.4f, SE: %.4f, F1: %.4f, IOU: %.4f, DC: %.4f' % (
					epoch + 1, self.num_epochs, \
					epoch_loss, \
					acc, PC, SP, SE, F1, JS, DC))

			# 学习率从某一代开始下降
			if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
				lr -= (self.lr / float(self.num_epochs_decay))
				for param_group in self.optimizer.param_groups:
					param_group['lr'] = lr
				print('Decay learning rate to lr: {}.'.format(lr))

			'''验证'''
			self.unet.train(False)
			self.unet.eval()

			acc = 0.  # Accuracy
			SE = 0.  # Sensitivity (Recall)
			SP = 0.  # Specificity
			PC = 0.  # Precision
			F1 = 0.  # F1 Score
			JS = 0.  # Jaccard Similarity
			DC = 0.  # Dice Coefficient
			length = 0
			for i, (images, GT) in enumerate(self.valid_loader):
				images = images.to(self.device)
				GT = GT.to(self.device)
				SR = F.sigmoid(self.unet(images))  # 这个其实可以写在模型里
				acc += get_accuracy(SR, GT)
				SE += get_sensitivity(SR, GT)
				SP += get_specificity(SR, GT)
				PC += get_precision(SR, GT)
				F1 += get_F1(SR, GT)
				JS += get_JS(SR, GT)
				DC += get_DC(SR, GT)
				length += 1

			acc = acc / length
			SE = SE / length
			SP = SP / length
			PC = PC / length
			F1 = F1 / length
			JS = JS / length
			DC = DC / length
			unet_score = JS + DC

			print('[Validation] Acc: %.4f, PC: %.4f, SP: %.4f, SE: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
				acc, PC, SP, SE, F1, JS, DC))

			'''
            torchvision.utils.save_image(images.data.cpu(),
                                        os.path.join(self.result_path,
                                                    '%s_valid_%d_image.png'%(self.model_type,epoch+1)))
            torchvision.utils.save_image(SR.data.cpu(),
                                        os.path.join(self.result_path,
                                                    '%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
            torchvision.utils.save_image(GT.data.cpu(),
                                        os.path.join(self.result_path,
                                                    '%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
            '''
			# 新加的：Save Best U-Net model
			new_iou = JS
			if new_iou >= best_iou:
				best_iou = new_iou
				best_epoch = epoch
				best_unet = self.unet.state_dict()
				print('Best %s Model IOU : %.4f; Best epoch : %d' % (self.model_type, best_iou, best_epoch))
				torch.save(best_unet, unet_path)

			# Save Best U-Net model
			# if unet_score > best_unet_score:
			# 	best_unet_score = unet_score
			# 	best_epoch = epoch
			# 	best_unet = self.unet.state_dict()
			# 	print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
			# 	torch.save(best_unet,unet_path)

		# 可视化模块关闭
		vis.close_summary()



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

		acc = 0.  # Accuracy
		SE = 0.  # Sensitivity (Recall)
		SP = 0.  # Specificity
		PC = 0.  # Precision
		F1 = 0.  # F1 Score
		JS = 0.  # Jaccard Similarity
		DC = 0.  # Dice Coefficient
		length = 0
		for i, (images, GT) in enumerate(self.test_loader):
			images = images.to(self.device)
			GT = GT.to(self.device)
			SR = F.sigmoid(self.unet(images))

			# 预测
			pre = SR
			pre[SR >= 0.5] = 255
			pre[SR < 0.5] = 0
			pre = pre.data
			# 把GPU上的数据扒下来放在CPU上才能处理
			pre = pre.cpu().numpy()[0]
			pre = np.squeeze(pre).astype(np.uint8)
			# pre = np.transpose(pre, (1, 2, 0))
			# filename = readfilename("path_image")
			filename = i
			filename = str(filename)
			tif.imwrite(self.result_path+ "/" + filename + "_pre.tif", pre)

			pre = GT * 255
			pre = pre.cpu().numpy()[0]
			pre = np.squeeze(pre).astype(np.uint8)
			tif.imwrite(self.result_path + "/" + filename + "_lab.tif", pre)

			pre = images * 255
			pre = pre.cpu().numpy()[0]
			pre = np.squeeze(pre).astype(np.uint8)
			pre = np.transpose(pre, (1, 2, 0))
			tif.imwrite(self.result_path + "/" + filename + "_img.tif", pre)
			print(self.result_path + "/" + filename + "_img" )

			acc += get_accuracy(SR, GT)
			SE += get_sensitivity(SR, GT)
			SP += get_specificity(SR, GT)
			PC += get_precision(SR, GT)
			F1 += get_F1(SR, GT)
			JS += get_JS(SR, GT)
			DC += get_DC(SR, GT)
			length += 1

		acc = acc / length
		SE = SE / length
		SP = SP / length
		PC = PC / length
		F1 = F1 / length
		JS = JS / length
		DC = DC / length
		unet_score = JS + DC

		# Print the log info
		print(
			'acc: %.4f, PC: %.4f, SP: %.4f, SE: %.4f, F1: %.4f, IOU: %.4f, DC: %.4f' % (
				acc, PC, SP, SE, F1, JS, DC))

		f = open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
		wr = csv.writer(f)
		wr.writerow(
			[self.model_type, acc, PC, SP, SE, F1, JS, DC, self.lr, self.num_epochs, self.num_epochs_decay,
			 self.augmentation_prob])
		f.close()

			
