import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastargs import get_current_config
from ffcv.fields import RGBImageField, IntField
from ffcv.writer import DatasetWriter
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Callable, Optional


class INaturalist(Dataset):
    def __init__(self, root: str, annotation: str, version: str="2018", transform: Optional[Callable]=None,
                 target_transform: Optional[Callable]=None) -> None:
        
        self.root = root
        self.version = version
        self.transform = transform
        self.target_transform = target_transform

        self.class_cnt = 8142  # Class count of inat18

        self.to_tensor = transforms.ToTensor()

        with open(annotation) as f:
            ann_data = json.load(f)

        self.imgs = [a["file_name"] for a in ann_data["images"]]

        if "annotations" in ann_data.keys():
            self.classes = [a["category_id"] for a in ann_data["annotations"]]
        else:
            self.calsses = [0] * len(self.imgs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        if self.root.endswith("/"):
            path = self.root + self.imgs[idx]
        else:
            path = self.root + "/" + self.imgs[idx]

        img = Image.open(path).convert("RGB")

        target = self.classes[idx]

        if self.transform is not None:
            img = self.transform(img)
        # TODO: Transform to tensor or not?
        #else:
        #    img = self.to_tensor(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def get_cls_cnt_list(self):
        cls_cnt_list = [0] * self.class_cnt

        for i in self.classes: cls_cnt_list[i] += 1

        return cls_cnt_list


def main():
    inat18_root = "/home/ktezoren/bot-lt/dsets/inat18"
    write_path = '/home/ktezoren/ffcv/beton-dsets'
    
    datasets = {
            "train": INaturalist(root=inat18_root, annotation=inat18_root + "/train2018.json"),
            "val": INaturalist(root=inat18_root, annotation=inat18_root + "/val2018.json")
            }


    for (name, ds) in datasets.items():
        writer = DatasetWriter(
                write_path + ("/inat18_train.beton" if name == "train" else "/inat18_val.beton"),
                {
                    'image': RGBImageField(max_resolution = 256),
                    'label': IntField()
                }
            )
        
        writer.from_indexed_dataset(ds)


if __name__ == "__main__":
    config = get_current_config()

    config.validate(mode="stderr")
    config.summary()

    main()

