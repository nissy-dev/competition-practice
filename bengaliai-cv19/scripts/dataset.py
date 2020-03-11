import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


HEIGHT = 137
WIDTH = 236
SIZE = 224


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


# https://www.kaggle.com/iafoss/image-preprocessing-128x128
def crop_resize(img, size=SIZE, pad=16):
    # crop a box around pixels large than the threshold
    # some images contain line at the sides
    ymin, ymax, xmin, xmax = bbox(img[5:-5, 5:-5] > 80)
    # cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img[ymin:ymax, xmin:xmax]
    # remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin, ymax-ymin
    l = max(lx, ly) + pad  # noqa
    # make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img, (size, size))


class BengaliAIDataset(Dataset):
    def __init__(self, images=None, labels=None, size=None, transforms=None):
        self.images = images
        self.labels = labels
        self.size = size
        self.transforms = transforms

        # set dummy labels
        if self.labels is None:
            self.labels = np.zeros(len(images))

        # validation
        if len(images) != len(labels):
            raise ValueError('Do not match the data size between input and output')

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.size is not None:
            img = crop_resize(img, self.size)
        # convert 3 channels
        img = np.stack([img, img, img]).transpose(1, 2, 0)
        if self.transforms is not None:
            augmented = self.transforms(image=img)
            img = augmented['image']
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return torch.tensor(img, dtype=torch.float), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.images)


# collate_fn
def mixup_collate_fn(batch, alpha=0.4):
    images, labels = list(zip(*batch))
    if np.random.rand() < 0.25:
        data = torch.stack(images)
        targets = torch.stack(labels)
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]
        shuffled_targets = targets[indices]
        lam = np.random.beta(alpha, alpha)
        data = data * lam + shuffled_data * (1 - lam)
        targets = [targets, shuffled_targets, lam]
        return data, targets
    else:
        images = torch.stack(images)
        labels = torch.stack(labels)
        return images, labels
