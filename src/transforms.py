from torchvision import transforms


def crop_and_flip():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def train_augm():
    return crop_and_flip()


def valid_augm():
    return crop_and_flip()
