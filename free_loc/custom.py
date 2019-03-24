import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

from PIL import Image
import os
import os.path
import numpy as np

import sklearn.metrics

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(imdb):
    #TODO: classes: list of classes
    #TODO: class_to_idx: dictionary with keys=classes and values=class index
    #If you did Task 0, you should know how to set these values from the imdb
    classes = imdb.classes
    class_to_idx = imdb._class_to_ind
    return classes, class_to_idx


def make_dataset(imdb, class_to_idx):
    #TODO: return list of (image path, list(+ve class indices)) tuples
    #You will be using this in IMDBDataset
    dataset_list = []
    im_indcies = imdb._load_image_set_index()
    for img_num, img_index in enumerate(im_indcies):
        img_path = imdb.image_path_at(img_num)
        class_indices = np.zeros(imdb.num_classes)
        class_indices[imdb._load_pascal_annotation(img_index)['gt_classes']-1]=1
        img_class_indices = class_indices
        dataset_list.append((img_path,img_class_indices))
    return dataset_list


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


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



class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()

        # # Features
        # self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=11,stride=4,padding=2)
        # self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,dilation=1)
        # self.conv2 = nn.Conv2d(in_channels=64,out_channels=192,kernel_size=5,stride=1,padding=1)
        # self.conv3 = nn.Conv2d(in_channels=192,out_channels=384,kernel_size=3,stride=1,padding=1)
        # self.conv4 = nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1)
        # self.conv5 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        # # Classification
        # self.conv6 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1)
        # self.conv7 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1,stride=1)
        # self.conv8 = nn.Conv2d(in_channels=256,out_channels=20,kernel_size=1,stride=1)

        #TODO: Define model
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, dilation=1, ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
            # nn.MaxPool2d(kernel_size=3, stride=2, dilation=1),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 20, kernel_size=3, stride=1, padding=1)
        )


    def forward(self, x):
        #TODO: Define forward pass
        x = self.features(x)
        x = self.classifier(x)
        return x

        # out = F.relu(self.conv1(x))
        # out = self.maxpool(out)

        # out = F.relu(self.conv2(x))
        # out = self.maxpool(out)

        # out = F.relu(self.conv3(x))
        # out = F.relu(self.conv4(x))
        # out = F.relu(self.conv5(x))
        # out = F.relu(self.conv6(x))
        # out = F.relu(self.conv7(x))
        # out = self.conv8(x)
        # return out




class LocalizerAlexNetHighres(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetHighres, self).__init__()
        #TODO: Ignore for now until instructed









    def forward(self, x):
        #TODO: Ignore for now until instructed










        return x

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def localizer_alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(**kwargs)
    model.features.apply(init_weights)
    model.classifier.apply(init_weights)
    model_state = model.state_dict()
    #TODO: Initialize weights correctly based on whethet it is pretrained or
    # not
    if(1):
        state_dict = model_zoo.load_url(model_urls['alexnet'])
        model_state['features.0.weight'] = state_dict['features.0.weight']
        model_state['features.0.bias'] = state_dict['features.0.bias']

        model_state['features.3.weight'] = state_dict['features.3.weight']
        model_state['features.3.bias'] = state_dict['features.3.bias']

        model_state['features.6.weight'] = state_dict['features.6.weight']
        model_state['features.6.bias'] = state_dict['features.6.bias']

        model_state['features.8.weight'] = state_dict['features.8.weight']
        model_state['features.8.bias'] = state_dict['features.8.bias']

        model_state['features.10.weight'] = state_dict['features.10.weight']
        model_state['features.10.bias'] = state_dict['features.10.bias']

        model.load_state_dict(model_state)
    return model





def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(**kwargs)
    #TODO: Ignore for now until instructed








    return model

def compute_ap(gt, pred, valid, average=None):
    """
    Compute the multi-label classification accuracy.
    Args:
        gt (np.ndarray): Shape Nx20, 0 or 1, 1 if the object i is present in that
            image.
        pred (np.ndarray): Shape Nx20, probability of that object in the image
            (output probablitiy).
        valid (np.ndarray): Shape Nx20, 0 if you want to ignore that class for that
            image. Some objects are labeled as ambiguous.
    Returns:
        AP (list): average precision for all classes
    """
    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()
    # valid = valid.data.numpy()
    nclasses = gt.shape[1]
    AP = []
    for cid in range(nclasses):
        gt_cls = gt[:, cid][(valid[:, cid] > 0).astype(int)].astype('float32')
        pred_cls = pred[:, cid][(valid[:, cid] > 0).astype(int)].astype('float32')
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        pred_cls -= 1e-5 * gt_cls
        ap = sklearn.metrics.average_precision_score(
            gt_cls, pred_cls, average=average)
        AP.append(ap)
    return AP


class IMDBDataset(data.Dataset):
    """A dataloader that reads imagesfrom imdbs
    Args:
        imdb (object): IMDB from fast-rcnn repository
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, list(+ve class indices)) tuples
    """

    def __init__(self, imdb, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(imdb)
        imgs = make_dataset(imdb, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images, what's going on?"))
        self.imdb = imdb
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
            tuple: (image, target) where target is a binary vector with 1s
                                   for +ve classes and 0s for -ve classes
                                   (it can be a numpy array)
        """
        # TODO: Write this function, look at the imagenet code for inspiration
        image_name, target = self.imgs[index]
        img = self.loader(image_name)
        img = self.transform(img).float()
        # img = (Image.open(image_name), dtype=np.float32)
        # target = np.zeros(len(self.classes))
        # target[self.imdb._load_pascal_annotation(self.imdb.image_index[index])['gt_classes']-1]=1
        # target[imdb._load_pascal_annotation(imdb.image_index[index])['gt_classes']-1]=1
        return img, target

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
