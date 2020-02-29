import os
import random
from torch.utils import data
from torchvision import transforms as T
from PIL import Image
from data_processing import writefilename
import auto_augment_improve


class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=512, mode='train', augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        self.root = root

        # GT : Ground Truth
        self.GT_paths = root[:-1] + '_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob
        self.output_ch = 16
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

        if self.mode == 'test':
            '''arrange'''
            self.image_paths = sorted(self.image_paths)

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        filename = image_path.split('/')[-1][:-len(".tif")]
        # GT_path = self.GT_paths + filename + '.tif'


        image = Image.open(image_path)
        # GT = Image.open(GT_path)
        # GT = GT.convert('L')
        # GT = GT.convert('RGB')

        Transform = []

        p_transform = random.random()

        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            '''rotation'''
            RotationDegree = random.randint(0, 3)
            RotationDegree = self.RotationDegree[RotationDegree]

            Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))

            Transform = T.Compose(Transform)

            # GT = Transform(GT)
            image = Transform(image)

            Transform = []

        # if (self.mode == 'train') and p_transform <= self.augmentation_prob:
        #     '''auto augment'''
        #     image, GT = auto_augment_improve.apply(image, GT)

        # GT = GT.convert('L')
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)

        image = Transform(image)
        # GT = Transform(GT)

        return image,filename + '.tif'

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)


def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train', augmentation_prob=0.4):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)
    # len = dataset.__len__()

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
