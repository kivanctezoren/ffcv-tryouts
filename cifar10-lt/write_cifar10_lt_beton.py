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
from bot_lt_transform_wrapper import TRANSFORMS
import numpy as np
from utils.utils import get_category_list
import math
from PIL import Image

class BaseSet(Dataset):
    def __init__(self, mode="train"):
        self.mode = mode
        self.input_size = (32,32)
        self.color_space = 'RGB'
        self.size = self.input_size

        print("Use {} Mode to train network".format(self.color_space))


        if self.mode == "train":
            print("Loading train data ...", end=" ")
            self.json_path = '/home/ktezoren/bot-lt/dsets/cifar-lt/converted/cifar10_imbalance50/cifar10_imbalance50_train.json'
        elif "valid" in self.mode:
            print("Loading valid data ...", end=" ")
            self.json_path = '/home/ktezoren/bot-lt/dsets/cifar-lt/converted/cifar10_imbalance50/cifar10_imbalance50_valid.json'

        with open(self.json_path, "r") as f:
            self.all_info = json.load(f)
        self.num_classes = self.all_info["num_classes"]

        if not self.cfg.DATASET.USE_CAM_BASED_DATASET or self.mode != 'train':
            self.data = self.all_info['annotations']
        else:
            assert os.path.isfile(self.cfg.DATASET.CAM_DATA_JSON_SAVE_PATH), \
                'the CAM-based generated json file does not exist!'
            self.data = json.load(open(self.cfg.DATASET.CAM_DATA_JSON_SAVE_PATH))
        print("Contain {} images of {} classes".format(len(self.data), self.num_classes))

    def update(self, epoch):
        self.epoch = max(0, epoch-self.cfg.TRAIN.TWO_STAGE.START_EPOCH) if self.cfg.TRAIN.TWO_STAGE.DRS else epoch
        if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "progressive":
            self.progress_p = epoch/self.cfg.TRAIN.MAX_EPOCH * self.class_p + (1-epoch/self.cfg.TRAIN.MAX_EPOCH)*self.instance_p
            print('self.progress_p', self.progress_p)


    def __getitem__(self, index):
        print('start get item...')
        now_info = self.data[index]
        img = self._get_image(now_info)
        print('complete get img...')
        meta = dict()
        image = self.transform(img)
        image_label = (
            now_info["category_id"] if "test" not in self.mode else 0
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
        imgage = self._get_image(now_info)
        image_label = now_info['category_id']  # 0-index
        
        return image, image_label


def main(train_dataset, val_dataset):

    cifar_json_path = "~/bot-lt/dsets/cifar-lt/converted/cifar10_imbalance50/"
    train_json_path = cifar_json_path + "cifar10_imbalance50_train.json"
    valid_json_path = cifar_json_path + "cifar10_imbalance50_valid.json"

    datasets = {
        'train': CIFAR('train'),
        'test': CIFAR('test')
        }

    for (name, ds) in datasets.items():
        path = '../beton-dstest/cifar10_imbalance50_train.beton' if name == 'train' else '../beton-dsets/cifar10_imbalance50_train.beton'
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
