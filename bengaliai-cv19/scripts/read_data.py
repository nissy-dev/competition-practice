import gc
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_data(BASE_PATH):
    print('Reading train.csv file....')
    train = pd.read_csv(BASE_PATH + 'train.csv')
    print('Training.csv file have {} rows and {} columns'.format(
        train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv(BASE_PATH + 'test.csv')
    print('Test.csv file have {} rows and {} columns'.format(
        test.shape[0], test.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv(BASE_PATH + 'sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(
        sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, sample_submission


def prepare_image(datadir, data_type='train', submission=False):
    assert data_type in ['train', 'test']
    images = []
    for i in tqdm([0, 1, 2, 3]):
        if submission:
            image_df_list = pd.read_parquet(datadir + f'{data_type}_image_data_{i}.parquet')
        else:
            image_df_list = pd.read_feather(datadir + f'{data_type}_image_data_{i}.feather')

        HEIGHT = 137
        WIDTH = 236
        image = image_df_list.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)
        images.append(image)
        del image_df_list, image
        gc.collect()

    images = np.concatenate(images, axis=0)
    print('Image shape : ', images.shape)
    return images
