# TODO: Not all modules may be needed

from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm

import torch as ch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
import torchvision

from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter


Section('training', 'Hyperparameters').params(
    lr=Param(float, 'The learning rate to use', required=True),
    epochs=Param(int, 'Number of epochs to run for', required=True),
    batch_size=Param(int, 'Batch size', default=512),
    momentum=Param(float, 'Momentum for SGD', default=0.9),
    weight_decay=Param(float, 'l2 weight decay', default=5e-4),
    num_workers=Param(int, 'The number of workers', default=8),
    lr_tta=Param(bool, 'Test time augmentation by averaging with horizontally flipped version', default=True)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    val_dataset=Param(str, '.dat file to use for validation', required=True),
)


@param('data.train_dataset')
@param('data.val_dataset')
@param('training.batch_size')
@param('training.num_workers')
def make_dataloaders(train_dataset=None, val_dataset=None, batch_size=None, num_workers=None):
    paths = {
        'train': train_dataset,
        'test': val_dataset
    }

    start_time = time.time()
    loaders = {}

    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]

        if name == 'train':
            image_pipeline: List[Operation] = [
                RandomResizedCropRGBImageDecoder(output_size=[224,224]),  # Default resizing arguments apply
                RandomHorizontalFlip()
            ]
        else:  # test
            image_pipeline: List[Operation] = [
                # FIXME: Doesn't fully replicate BoT config.: Also need to resize with BoT transformation
                #   "shorter_resize_for_crop"
                CenterCropRGBImageDecoder(output_size=[224,224], ratio=1)
            ]

        # (Leave out normalization with mean & std. to match BoT config.)
        image_pipeline.extend([
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16)  # TODO: float16 or 32?
        ])
        
        # Shuffle if train, do not if validation
        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name], batch_size=batch_size, num_workers=num_workers,
                               order=ordering, drop_last=(name == 'train'),
                               pipelines={'image': image_pipeline, 'label': label_pipeline})

    return loaders, start_time

# Model (from KakaoBrain: https://github.com/wbaek/torchskeleton)
class Mul(ch.nn.Module):
    def __init__(self, weight):
       super(Mul, self).__init__()
       self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(ch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(ch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return ch.nn.Sequential(
            ch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                         stride=stride, padding=padding, groups=groups, bias=False),
            ch.nn.BatchNorm2d(channels_out),
            ch.nn.ReLU(inplace=True)
    )


class ResNet(ch.nn.Module):
    def __init__(self, block_type, num_blocks, last_layer_stride=2):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.block = block_type
        self.conv1 = ch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = ch.nn.BatchNorm2d(64)
        self.relu = ch.nn.ReLU(True)
        self.pool = ch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(num_blocks[0], 64)
        self.layer2 = self._make_layer(
            num_blocks[1], 128, stride=2
        )
        self.layer3 = self._make_layer(
            num_blocks[2], 256, stride=2
        )
        self.layer4 = self._make_layer(
            num_blocks[3],
            512,
            stride=last_layer_stride,
        )
    def _make_layer(self, num_block, planes, stride=1):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for now_stride in strides:
            layers.append(
                self.block(
                    self.inplanes, planes, stride=now_stride
                )
            )
            self.inplanes = planes * self.block.expansion
        return ch.nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.layer1(out)
        if 'layer' in kwargs and kwargs['layer'] == 'layer1':
            out = kwargs['coef']*out + (1-kwargs['coef'])*out[kwargs['index']]
        out = self.layer2(out)
        if 'layer' in kwargs and kwargs['layer'] == 'layer2':
            out = kwargs['coef']*out+(1-kwargs['coef'])*out[kwargs['index']]
        out = self.layer3(out)
        if 'layer' in kwargs and kwargs['layer'] == 'layer3':
            out = kwargs['coef']*out+(1-kwargs['coef'])*out[kwargs['index']]
        out = self.layer4(out)
        if 'layer' in kwargs and kwargs['layer'] == 'layer4':
            out = kwargs['coef']*out+(1-kwargs['coef'])*out[kwargs['index']]
        return out


class BottleNeck(ch.nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = ch.nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = ch.nn.BatchNorm2d(planes)
        self.relu1 = ch.nn.ReLU(True)
        self.conv2 = ch.nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = ch.nn.BatchNorm2d(planes)
        self.relu2 = ch.nn.ReLU(True)
        self.conv3 = ch.nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = ch.nn.BatchNorm2d(planes * self.expansion)
        if stride != 1 or self.expansion * planes != inplanes:
            self.downsample = ch.nn.Sequential(
                ch.nn.Conv2d(
                    inplanes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                ch.nn.BatchNorm2d(self.expansion * planes),
            )
        else:
            self.downsample = None
        self.relu = ch.nn.ReLU(True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))

        out = self.relu2(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))

        if self.downsample != None:
            residual = self.downsample(x)
        else:
            residual = x
        out = out + residual
        out = self.relu(out)
        return out


def rn_50(last_layer_stride=2):
    return ResNet(BottleNeck, [3, 4, 6, 3], last_layer_stride)


class GAP(ch.nn.Module):
    """Global Average pooling
        Widely used in ResNet, Inception, DenseNet, etc.
     """

    def __init__(self):
        super(GAP, self).__init__()
        self.avgpool = ch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.avgpool(x)
        #         x = x.view(x.shape[0], -1)
        return x


class Network(ch.nn.Module):
    def __init__(self, mode="train", num_classes=8142):
        super(Network, self).__init__()
        self.num_classes = num_classes
        self.backbone = rn_50(last_layer_stride=2)

        self.mode = mode
        self.module = GAP()
        self.classifier = ch.nn.Linear(2048, self.num_classes, bias=True)

    def forward(self, x, **kwargs):
        if "feature_flag" in kwargs or "feature_cb" in kwargs or "feature_rb" in kwargs:
            return self.extract_feature(x, **kwargs)
        elif "classifier_flag" in kwargs:
            return self.classifier(x)
        elif 'feature_maps_flag' in kwargs:
            return self.extract_feature_maps(x)
        elif 'layer' in kwargs and 'index' in kwargs:
            if kwargs['layer'] in ['layer1', 'layer2', 'layer3']:
                x = self.backbone.forward(x, index=kwargs['index'], layer=kwargs['layer'], coef=kwargs['coef'])
            else:
                x = self.backbone(x)
            x = self.module(x)
            if kwargs['layer'] == 'pool':
                x = kwargs['coef']*x+(1-kwargs['coef'])*x[kwargs['index']]
            x = x.view(x.shape[0], -1)
            x = self.classifier(x)
            if kwargs['layer'] == 'fc':
                x = kwargs['coef']*x + (1-kwargs['coef'])*x[kwargs['index']]
            return x

        x = self.backbone(x)
        x = self.module(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def get_backbone_layer_info(self):
        layers = 4
        blocks_info = [3, 4, 6, 3]
        return layers, blocks_info

    def extract_feature(self, x, **kwargs):
        x = self.backbone(x)
        x = self.module(x)
        x = x.view(x.shape[0], -1)
        return x

    def extract_feature_maps(self, x):
        x = self.backbone(x)
        return x

    def freeze_backbone(self):
        print("Freezing backbone .......")
        for p in self.backbone.parameters():
            p.requires_grad = False

    def get_feature_length(self):
        return 2048

    def _get_module(self):
        return GAP()


def construct_model():
    num_classes = 8142

    # Construct ResNet50
    model = Network("train", num_classes)

    """
    model = ch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(ch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        ch.nn.MaxPool2d(2),
        Residual(ch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        ch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        ch.nn.Linear(128, num_class, bias=False),
        Mul(0.2)
    )
    """

    model = model.to(memory_format=ch.channels_last).cuda()
    return model


@param('training.lr')
@param('training.epochs')
@param('training.momentum')
@param('training.weight_decay')
def train(model, loaders, lr=None, epochs=None, label_smoothing=None,
          momentum=None, weight_decay=None, lr_peak_epoch=None):
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loaders['train'])
    scheduler = lr_scheduler.MultiStepLR(opt, [60, 80], 0.1)
    scaler = GradScaler(init_scale=16384.0)
    loss_fn = CrossEntropyLoss()

    for _ in range(epochs):
        for ims, labs in tqdm(loaders['train']):
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

@param('training.lr_tta')
def evaluate(model, loaders, lr_tta=False):
    model.eval()
    with ch.no_grad():
        for name in ['train', 'test']:
            total_correct, total_num = 0., 0.
            for ims, labs in tqdm(loaders[name]):
                with autocast():
                    out = model(ims)
                    if lr_tta:
                        out += model(ch.fliplr(ims))
                    total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                    total_num += ims.shape[0]
            print(f'{name} accuracy: {total_correct / total_num * 100:.1f}%')

if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    # Also loads from args.config_path if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    loaders, start_time = make_dataloaders()
    model = construct_model()
    train(model, loaders)
    print(f'Total time: {time.time() - start_time:.5f}')
    evaluate(model, loaders)

