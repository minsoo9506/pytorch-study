from torchvision import transforms

class ImageTransform:
    """transform image with train, validation data

    Attributes:
    -----------
    resize : int
    mean : (R,G,B)
    std : (R,G,B)
    """

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train' : transforms.Compose([
                transforms.RandomResizedCrop(
                    resize, scale=(0.5, 1.0)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val' : transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, mode='train'):
        return self.data_transform[mode](img)

import os, glob

def make_datapath_list(mode='train'):
    """데이터 경로를 저장한 리스트 작성

    Parameters:
    -----------
    mode : 'train' or 'val'

    Returns:
    --------
    path_list : list
        데이터 경로를 저장한 리스트
    """
    root_path = './data/hymenoptera_data/'
    target_path = os.path.join(root_path, mode, '/**/*.jpg')

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list