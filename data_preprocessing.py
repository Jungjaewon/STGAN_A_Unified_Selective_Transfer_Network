import glob
from tqdm import tqdm
import random
import shutil
import os.path as osp
import os

def filtering_data(annotation_txt, selected_attr, img_dir, dst_dir):
    data_dict = dict()
    idx2attr = dict()
    attr2idx = dict()
    data_list = glob.glob(osp.join(img_dir , '*.jpg'))
    data_list = [x.split(os.sep)[-1] for x in data_list]

    all_lines = open(annotation_txt, 'r').readlines()[1:]
    attr_list = all_lines[0].split(' ')

    for idx, attr in enumerate(attr_list):
        attr = attr.strip()
        idx2attr[idx] = attr
        attr2idx[attr] = idx

    all_lines = all_lines[1:]

    for line in tqdm(all_lines):
        attribute = list()
        splited = line.split(' ')
        filename = splited[0]
        attr_list = splited[1:]

        for sel_attr in selected_attr:
            idx = attr2idx[sel_attr]
            if attr_list[idx] == '1':
                attribute.append(1)
            else:
                attribute.append(0)

        if sum(attribute) == 0:
            continue

        data_dict[filename] = attribute

    valid_list = list(data_dict.keys())
    data_list = set(valid_list).intersection(set(data_list))
    data_list = [osp.join(img_dir, x) for x in data_list]
    random.seed(349)
    random.shuffle(data_list)
    data_list = data_list[:20000]

    for idx, image_path in tqdm(enumerate(data_list)):
        image_name = image_path.split(os.sep)[-1]
        src = image_path
        dst = osp.join(dst_dir, image_name)
        shutil.copy(src, dst)



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
    #data_split('images')

    selected_attr = ['Brown_Hair', 'Smiling', 'Rosy_Cheeks', 'Eyeglasses', 'Double_Chin', 'Pale_Skin', 'Chubby', 'Bald',
                     'Heavy_Makeup', 'Male']
    annotation_txt = osp.join('celeba', 'list_attr_celeba.txt')
    image_dir = osp.join('celeba', 'images')
    filtering_data(annotation_txt, selected_attr, image_dir, 'train_data')




