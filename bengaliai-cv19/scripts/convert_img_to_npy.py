import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


DATA_PATH = ''


# 1枚ごとにデータを読み込むようにするためにnpyで保存
def convert_img_to_npy(img_ids, images, save_dir=DATA_PATH + 'img_npy'):
    assert len(img_ids) == len(images)
    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm(range(len(img_ids))):
        save_path = os.path.join(save_dir, img_ids[i])
        np.save('{}.npy'.format(save_path), images[i])
    return True


def img_to_npy(id, img, save_dir):
    save_path = os.path.join(save_dir, id)
    np.save('{}.npy'.format(save_path), img)
    return True


# 並列化バージョン
def convert_img_to_npy(img_ids, images, save_dir=DATA_PATH + 'train_img_npy'):  # noqa
    assert len(img_ids) == len(images)
    os.makedirs(save_dir, exist_ok=True)
    _ = Parallel(n_jobs=8)([
        delayed(img_to_npy)(img_ids[i], images[i], save_dir) for i in range(len(images))
    ])
    return True


# ファイルが正しく生成できているかチェック
def check_all_file(save_dir=DATA_PATH + 'train_img_npy'):
    NUM_ALL_FILE = 200840
    for i in range(NUM_ALL_FILE):
        file_name = 'Train_{}.npy'.format(i)
        path = os.path.join(save_dir, file_name)
        if os.path.isfile(path) is False:
            raise ValueError('File {} not found...'.format(file_name))
    return True
