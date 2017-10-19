import torch.utils.data as data
import pandas as pd
from PIL import Image
import os


def load(image, train=True):
    folder = "train" if train else "test"
    img = Image.open(f"../data/images/{folder}/{image}.jpg")
    return img

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
        df = df[df.image_id.str.contains("_0")]
        self.num_classes = labels_map.shape[0]
        self.labels_map = labels_map
        self.images = df.image_id.values
        self.labels = df.category_id.values
        self.transform = transform

    def __len__(self):
        return self.images.size

    def __getitem__(self, idx):
        X = load(self.images[idx])
        if self.transform:
            X = self.transform(X)
        y = self.labels_map.loc[self.labels[idx]].label_id
        return X, y


class TestDataset(data.Dataset):
    def __init__(self, transform):
        self.images = list(filter(lambda x: "_0" in x, os.listdir("../data/images/test")))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx].split(".")[0]
        img = load(img_name, train=False)
        img = self.transform(img)
        return img, img_name.split("_")[0]


class ValidDataset(data.Dataset):
    def __init__(self, n_fold, transform=None):
        df = pd.read_csv(f"../data/fold_{n_fold}.csv")
        df = df[df.image_id.str.contains("_0")]
        self.images = df.image_id.values
        self.product_ids = df.product_id.values
        self.transform = transform

    def __len__(self):
        return self.images.size

    def __getitem__(self, idx):
        X = load(self.images[idx])
        if self.transform:
            X = self.transform(X)
        y = self.product_ids[idx]
        return X, y


def get_test_loader(batch_size, transform):
    test_dataset = TestDataset(transform)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=6)
    return test_loader


def get_valid_loader(n_fold, batch_size, transform):
    dataset = ValidDataset(n_fold, transform)
    dataset = data.DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=6)
    return dataset


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
