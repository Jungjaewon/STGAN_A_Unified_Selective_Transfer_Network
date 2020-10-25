import os
import os.path as osp
import glob
import torch

from torch.utils import data
from torchvision import transforms as T
from PIL import Image


class DataSet(data.Dataset):

    def __init__(self, config, img_transform):
        self.img_transform = img_transform
        self.img_dir = osp.join(config['TRAINING_CONFIG']['IMG_DIR'])
        self.img_size = (config['MODEL_CONFIG']['IMG_SIZE'], config['MODEL_CONFIG']['IMG_SIZE'], 3)
        self.annotation_txt = config['TRAINING_CONFIG']['ANNOTATION_TXT']
        self.selected_attr = ['Brown_Hair', 'Smiling', 'Rosy_Cheeks', 'Eyeglasses', 'Double_Chin', 'Pale_Skin', 'Chubby', 'Bald', 'Heavy_Makeup', 'Male']
        self.data_list = glob.glob(osp.join(self.img_dir , '*.jpg'))
        self.data_list = [x.split(os.sep)[-1] for x in self.data_list]
        self.data_dict = dict()
        self.idx2attr = dict()
        self.attr2idx = dict()

        self.preprocess()

    def preprocess(self):
        print('Preprocessing...')
        all_lines = open(self.annotation_txt, 'r') .readlines()[1:]
        attr_list = all_lines[0].split(' ')

        for idx, attr in enumerate(attr_list):
            attr = attr.strip()
            self.idx2attr[idx] = attr
            self.attr2idx[attr] = idx

        all_lines = all_lines[1:]

        for line in all_lines:
            attribute = list()
            splited = line.split(' ')
            filename = splited[0]
            attr_list = splited[1:]

            for sel_attr in self.selected_attr:
                idx = self.attr2idx[sel_attr]
                if attr_list[idx] == '1':
                    attribute.append(1)
                else:
                    attribute.append(0)

            if sum(attribute) == 0:
                continue

            self.data_dict[filename] = attribute

        valid_list = list(self.data_dict.keys())
        self.data_list = set(valid_list).intersection(set(self.data_list))
        self.data_list = [osp.join(self.img_dir, x) for x in self.data_list]
        print('Preprocessing is finished')

    def __getitem__(self, index):
        image_path = self.data_list[index]
        image_name = image_path.split(os.sep)[-1]
        attribute = self.data_dict[image_name]
        image = Image.open(image_path).convert('RGB')
        return self.img_transform(image), torch.FloatTensor(attribute)

    def __len__(self):
        """Return the number of images."""
        return len(self.data_list)


def get_loader(config):

    img_transform = list()
    img_size = config['MODEL_CONFIG']['IMG_SIZE']

    img_transform.append(T.Resize((img_size, img_size)))
    img_transform.append(T.ToTensor())
    img_transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    img_transform = T.Compose(img_transform)

    dataset = DataSet(config, img_transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['TRAINING_CONFIG']['BATCH_SIZE'],
                                  shuffle=(config['TRAINING_CONFIG']['MODE'] == 'train'),
                                  num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                  drop_last=True)
    return data_loader
