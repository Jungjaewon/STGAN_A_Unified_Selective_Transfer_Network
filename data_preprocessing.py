import glob
from tqdm import tqdm
import random
import shutil
import os.path as osp
import os


def distribute(data_list, mode='train'):
    assert mode in ['train', 'test']
    for idx, image_path in tqdm(enumerate(data_list)):
        image_name = image_path.split(os.sep)[-1]
        src = image_path
        dst = osp.join(mode, image_name)
        shutil.copy(src, dst)



def data_split(image_folder):
    image_list = glob.glob(osp.join(image_folder, '*.jpg'))
    random.seed(3534)
    random.shuffle(image_list)
    cut = len(image_list)
    train_list = image_list[: int(cut * 0.9)]
    test_list = image_list[int(cut * 0.9):]
    distribute(train_list, mode='train')
    distribute(test_list, mode='test')


if __name__ == '__main__':
    pass
    """
    1. remove small dataset
    2. file name rename (class_idx)_idx.png
    3. intergrate all image into a folder
    """
    data_split('images')

