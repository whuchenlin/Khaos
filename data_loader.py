import os
import random
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
from data_processing import writefilename

class ImageFolder(data.Dataset):
	def __init__(self, root,image_size=512,mode='train',augmentation_prob=0.4):
		"""Initializes image paths and preprocessing module."""
		self.root = root
		
		# GT : Ground Truth
		self.GT_paths = root[:-1]+'_GT/'
		# map（）根据函数对指定的序列做映射 也就是：路径+图像名
		self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
		self.image_size = image_size
		self.mode = mode
		self.RotationDegree = [0,90,180,270]
		self.augmentation_prob = augmentation_prob
		self.output_ch = 16
		print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		image_path = self.image_paths[index]
		filename = image_path.split('/')[-1][:-len(".tif")]
		GT_path = self.GT_paths + filename + '.tif'

		# # 测试时，保存影像路径
		if self.mode == 'test':
			writefilename('path_image', filename + '.tif')
		# 	writefilename('path_label', filename + '.tif')

		image = Image.open(image_path)
		GT = Image.open(GT_path)
		GT = GT.convert('L')

		aspect_ratio = image.size[1]/image.size[0]

		Transform = []

        # 重设宽高？300-320之间
		# ResizeRange = random.randint(400,420)
		# Transform.append(T.Resize((int(ResizeRange*aspect_ratio),ResizeRange), interpolation=Image.NEAREST))
		p_transform = random.random()

		if (self.mode == 'train') and p_transform <= self.augmentation_prob:
			RotationDegree = random.randint(0,3)
			RotationDegree = self.RotationDegree[RotationDegree]
			if (RotationDegree == 90) or (RotationDegree == 270):
				aspect_ratio = 1/aspect_ratio

			# RandomRotation对图像进行任意角度旋转，锁死了，范围：(RotationDegree,RotationDegree)
			Transform.append(T.RandomRotation((RotationDegree,RotationDegree)))

			# RotationRange = random.randint(-10,10)
			# Transform.append(T.RandomRotation((RotationRange,RotationRange)))
			#
			# CropRange = random.randint(250,270)
			# Transform.append(T.CenterCrop((int(CropRange*aspect_ratio),CropRange)))
			Transform = T.Compose(Transform)

			GT = Transform(GT)

			# ShiftRange_left = random.randint(0,20)
			# ShiftRange_upper = random.randint(0,20)
			# ShiftRange_right = image.size[0] - random.randint(0,20)
			# ShiftRange_lower = image.size[1] - random.randint(0,20)
			# image = image.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))
			# GT = GT.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))

			# if random.random() < 0.5:
			# 	image = F.hflip(image)
			# 	GT = F.hflip(GT)
			#
			# if random.random() < 0.5:
			# 	image = F.vflip(image)
			# 	GT = F.vflip(GT)

			# .修改亮度、对比度和饱和度
			# Transform = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02)
			image = Transform(image)
			Transform = []

        # Transform.append(T.Resize((int(256*aspect_ratio)-int(256*aspect_ratio)%16,256)))
		# Transform.append(T.Resize((int(256 * aspect_ratio) - int(256 * aspect_ratio) % 16, 256)))
		Transform.append(T.ToTensor())
		Transform = T.Compose(Transform)

		image = Transform(image)
		GT = Transform(GT)

		return image, GT

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
	"""Builds and returns Dataloader."""
	
	dataset = ImageFolder(root = image_path, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
	len = dataset.__len__()

	if mode == 'train':
		data_loader = data.DataLoader(dataset=dataset,
									  batch_size=batch_size,
									  shuffle=True,
									  num_workers=num_workers)
	else:
		data_loader = data.DataLoader(dataset=dataset,
									  batch_size=batch_size,
									  shuffle=False,
									  num_workers=num_workers)

	return data_loader
