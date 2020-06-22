'''
code by zzg   2020-06-12
'''
try:
    import xml.etree.cElementTree as ET  
except ImportError:
    import xml.etree.ElementTree as ET

import torch.utils.data as data
 
from PIL import Image
import os
import os.path
import cv2
import torch
import numpy as np


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
 
 

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
 
 
def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
 
 
def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
 
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
 
    return images
 
 
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            # return img.convert('RGB')
            img = img.convert('RGB')

    path_xml = path.replace(".jpeg", ".xml")
    tree = ET.ElementTree(file=path_xml)
    root = tree.getroot()
    ObjectSet = root.findall('object')
    ObjBndBoxSet = []
    for Object in ObjectSet:
        # ObjName = Object.find('name').text
        BndBox = Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)#-1 
        y1 = int(BndBox.find('ymin').text)#-1
        x2 = int(BndBox.find('xmax').text)#-1
        y2 = int(BndBox.find('ymax').text)#-1
        BndBoxLoc = [x1,y1,x2,y2]
        # print(x1,y1,x2,y2)
        ObjBndBoxSet.append(BndBoxLoc) 
    x1,y1,x2,y2 = ObjBndBoxSet[0]
    # img1 = cv2.rectangle(im,(x01,y01),(x02,y02),(255,0,0),2)
    
    img = np.array(img)
    img1 = img[y1:y2, x1:x2]

    out = torch.from_numpy(cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC))
    out1 = torch.from_numpy(cv2.resize(img1, (224, 224), interpolation=cv2.INTER_CUBIC))
    img = torch.cat((out,out1),1)
    img = img.numpy()
    img = Image.fromarray(img.astype('uint8')).convert('RGB')  ##numpy.array to PIL.Image
    return img
    # print(img.shape)

 
def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
 
 
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def parse_xml(path):
    tree = ET.ElementTree(file=path)
    root = tree.getroot()
    ObjectSet = root.findall('object')
    ObjBndBoxSet = []
    for Object in ObjectSet:
        ObjName = Object.find('name').text
        BndBox = Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)#-1 
        y1 = int(BndBox.find('ymin').text)#-1
        x2 = int(BndBox.find('xmax').text)#-1
        y2 = int(BndBox.find('ymax').text)#-1
        BndBoxLoc = [ObjName,x1,y1,x2,y2]
        ObjBndBoxSet.append(BndBoxLoc) 
    return ObjBndBoxSet



class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
 
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
 
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
 
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        # print(path)
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
 
        return img, target
 
 
    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":

    TRAIN_DATA_PATH = "/home/zigangzhao/DMS/pytorch-cifar100/data/dmsdata/train/"
    train_data = ImageFolder(root=TRAIN_DATA_PATH)
    print(train_data[0])