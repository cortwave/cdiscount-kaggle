from torchvision import transforms
from imgaug import augmenters as iaa


def normalize():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def flip_and_rotate():
    return iaa.Sequential([
        iaa.Fliplr(0.2),
        iaa.Affine(rotate=(-45, 45), mode="edge")
    ]).augment_image


def train_augm():
    return transforms.Compose([
        flip_and_rotate(),
        normalize()
    ])


def valid_augm():
    return normalize()
