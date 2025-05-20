import os
from PIL import Image
from scipy.io import loadmat

import torch

import torchvision.transforms as T
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DogHeartDataset(object):
    """
    DogHeart Dataset.
    Args:
        root (string): Root directory where the dataset is stored.
        transforms (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = [filepath for filepath in list(sorted(os.listdir(os.path.join(root, "Images")))) 
                     if filepath.endswith('png') or filepath.endswith('jpg')]
        self.points = list(sorted(os.listdir(os.path.join(root, "Labels"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        points_path = os.path.join(self.root, "Labels", self.points[idx])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if self.transforms is not None:
            img = self.transforms(img)
        h_new, w_new = img.shape[1], img.shape[2]
        mat = loadmat(points_path)
        six_points = mat['six_points'].astype(float)
        six_points = torch.as_tensor(six_points, dtype=torch.float32)
        six_points[:,0] = w_new / w * six_points[:,0] # Resize image to any size and maintain original points
        six_points[:,1] = h_new / h * six_points[:,1]
        six_points = torch.reshape(six_points, (-1,))/h_new # Normlize the points
        # six_points = torch.reshape(six_points, (-1,))
        VHS = mat['VHS'].astype(float)
        VHS = torch.as_tensor(VHS, dtype=torch.float32).reshape([1,1])
        return idx, img, six_points, VHS

    def __len__(self):
        return len(self.imgs)
    
class DogHeartTestDataset(torch.utils.data.Dataset):
    """
    DogHeart Test Dataset.
    Args:
        root (string): Root directory where the dataset is stored.
        transforms (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = [filepath for filepath in list(sorted(os.listdir(os.path.join(root, "Images")))) 
                     if filepath.endswith('png') or filepath.endswith('jpg')]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img, self.imgs[idx]

    def __len__(self):
        return len(self.imgs)


def get_transform(resized_image_size):
    """
    Get the transformation for the dataset.
    Args:
        resized_image_size (int): The size to which the image will be resized.
    Returns:
        T.Compose: A composed transform that includes resizing, normalization, and conversion to tensor.
    """
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.Resize((resized_image_size, resized_image_size)))
    transforms.append(T.Normalize(mean = [0.485,0.456,0.406], std = [0.229, 0.224, 0.225]))
    return T.Compose(transforms)

