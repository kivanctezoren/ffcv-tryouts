from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm

import torch as ch
import torchvision

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField

from torch.utils.data import Dataset
import torch
import json, os, random, time
import cv2
import torchvision.transforms as transforms
import numpy as np
import math
from PIL import Image


# From BoT's utils:
def get_category_list(annotations, num_classes):
    num_list = [0] * num_classes
    cat_list = []
    print("Weight List has been produced")
    for anno in annotations:
        category_id = anno["category_id"]
        num_list[category_id] += 1
        cat_list.append(category_id)
    return num_list, cat_list 


class BaseSet(Dataset):
    def __init__(self, mode="train", sampler_type="default", two_stage_training=False):
        self.mode = mode
        self.input_size = (32, 32)
        self.color_space = 'RGB'
        self.size = self.input_size

        print("Use {} Mode to train network".format(self.color_space))


        if self.mode == "train":
            print("Loading train data ...", end=" ")
            self.json_path = './jsons/im50/cifar10_imbalance50_train.json'
        else:  # valid
            print("Loading valid data ...", end=" ")
            self.json_path = './jsons/im50/cifar10_imbalance50_valid.json'

        with open(self.json_path, "r") as f:
            self.all_info = json.load(f)
        self.num_classes = self.all_info["num_classes"]

        """
        if not self.mode != 'train':
            self.data = self.all_info['annotations']
        else:
            assert os.path.isfile(self.cfg.DATASET.CAM_DATA_JSON_SAVE_PATH), \
                'the CAM-based generated json file does not exist!'
            self.data = json.load(open(self.cfg.DATASET.CAM_DATA_JSON_SAVE_PATH))
        """

        self.data = self.all_info['annotations']

        print("Contain {} images of {} classes".format(len(self.data), self.num_classes))

    def update(self, epoch):
        # TODO: Placeholder twostage_startepoch value used. Pass value as config. if two stage will be implemented
        twostage_startepoch = 30
        self.epoch = max(0, epoch - twostage_startepoch) if two_stage_training else epoch
        if self.sampler_type == "progressive":
            # TODO: Placeholder max_epoch value used. Pass max_epoch value as config. when sampler is being implemented
            max_epoch = 100
            self.progress_p = epoch/max_epoch * self.class_p + (1-epoch/max_epoch)*self.instance_p
            print('self.progress_p', self.progress_p)


    def __getitem__(self, index):
        #print('start get item...')
        now_info = self.data[index]
        img = self._get_image(now_info)
        #print('complete get img...')
        meta = dict()
        image = self.transform(img)
        image_label = (
            now_info["category_id"] if "valid" not in self.mode else 0
        )  # 0-index
        if self.mode not in ["train", "valid"]:
           meta["image_id"] = now_info["image_id"]
           meta["fpath"] = now_info["fpath"]

        return image, image_label, meta

    def get_annotations(self):
        return self.all_info['annotations']

    def __len__(self):
        return len(self.all_info['annotations'])

    def imread_with_retry(self, fpath):
        retry_time = 10
        for k in range(retry_time):
            try:
                img = cv2.imread(fpath)
                if img is None:
                    print(fpath)
                    print("img is None, try to re-read img")
                    continue
                return img
            except Exception as e:
                if k == retry_time - 1:
                    assert False, "pillow open {} failed".format(fpath)
                time.sleep(0.1)

    def _get_image(self, now_info):
        fpath = os.path.join(now_info["fpath"])
        img = self.imread_with_retry(fpath)
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.data):
            cat_id = (
                anno["category_id"] if "category_id" in anno else anno["image_label"]
            )
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict


class CIFAR(BaseSet):
    def __init__(self, mode = 'train'):
        super().__init__(mode)
        random.seed(0)
        self.class_dict = self._get_class_dict()


    def __getitem__(self, index):        
        now_info = self.data[index]
        image = self._get_image(now_info)
        image_label = now_info['category_id']  # 0-index
        
        return image, image_label


def main():
    #cifar_json_path = "./jsons/im50/"
    #train_json_path = cifar_json_path + "cifar10_imbalance50_train.json"
    #valid_json_path = cifar_json_path + "cifar10_imbalance50_valid.json"

    datasets = {
        'train': CIFAR('train'),
        'valid': CIFAR('valid')
        }

    for (name, ds) in datasets.items():
        # TODO: Support for different jsons with different imbalance settings?
        path = '../beton-dsets/cifar10_lt_im50_train.beton' if name == 'train' else '../beton-dsets/cifar10_lt_im50_valid.beton'
        writer = DatasetWriter(path, {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)


if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='CIFAR10_Imbalance50_beton')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    main()
