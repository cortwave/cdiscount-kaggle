import torch.utils.data as data
import pandas as pd
from skimage.io import imread
from skimage.transform import resize

class Dataset(data.Dataset):
    def __init__(self, n_fold, n_folds, transform=None, train=True):
        if train:
            folds = list(range(n_folds))
            folds.remove(n_fold)
            train_dfs = [pd.read_csv(f"../data/fold_{i}.csv") for i in folds]
            df = pd.concat(train_dfs)
        else:
            df = pd.read_csv(f"../data/fold_{n_fold}.csv")
        labels_map = pd.read_csv("../data/labels_map.csv")
        labels_map.index = labels_map.category_id
        self.num_classes = labels_map.shape[0]
        self.labels_map = labels_map
        self.images = df.image_id.values
        self.labels = df.category_id.values
        self.transform = transform

    def __len__(self):
        return self.images.size

    @staticmethod
    def _load(image):
        img = imread(f"../data/images/train/{image}.jpg")
        img = resize(img, (256, 256), mode='constant')
        return img

    def __getitem__(self, idx):
        X = self._load(self.images[idx])
        if self.transform:
            X = self.transform(X)
        y = self.labels_map.ix[self.labels[idx]].label_id
        return X, y


def get_loaders(batch_size,
                n_fold,
                train_transform=None,
                valid_transform=None,
                n_folds=5):
    train_dataset = Dataset(n_fold, n_folds, train_transform, train=True)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=6,
                                   pin_memory=True)

    valid_dataset = Dataset(n_fold, n_folds, valid_transform, train=False)
    valid_loader = data.DataLoader(valid_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=6,
                                   pin_memory=True)
    return train_loader, valid_loader, train_dataset.num_classes

