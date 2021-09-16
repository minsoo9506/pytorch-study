from torch.utils.data import Dataset
from PIL import Image

class HymenopteraDataset(Dataset):
    """
    Attributes:
    -----------
    file_list : list
    transform : object
    mode : 'train' or 'val'
    """

    def __init__(self, file_list, transform=None, mode='train'):
        self.file_list = file_list
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(
            img, self.mode
        )

        if self.mode == 'train':
            label = img_path[30:34]
        elif self.mode == 'val':
            label = img_path[28:32]

        if label == 'ants':
            label = 0
        elif label == 'bees':
            label = 1

        return img_transformed, label        