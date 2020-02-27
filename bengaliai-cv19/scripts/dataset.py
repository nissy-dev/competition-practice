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


# ただの
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
        img = cv2.cvtColor(img, cv2.CV_GRAY2BGR)
        if self.transforms is not None:
            augmented = self.transforms(image=img)
            img = augmented['image']
        return torch.tensor(img), torch.tensor(label)

    def __len__(self):
        return len(self.images)
