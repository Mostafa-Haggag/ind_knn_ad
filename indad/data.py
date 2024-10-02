import os
import tarfile
from os.path import isdir
from pathlib import Path

import wget
from PIL import Image
from torch import tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

DATASETS_PATH = Path("./datasets")
# these are the means of imagenet
IMAGENET_MEAN = tensor([0.485, 0.456, 0.406])
IMAGENET_STD = tensor([0.229, 0.224, 0.225])

MVTEC_CLASSES = {
    "bottle": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz",
    "cable": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951498/cable.tar.xz",
    "capsule": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz",
    "carpet": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz",
    "grid": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz",
    "hazelnut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545-1629951845/hazelnut.tar.xz",
    "leather": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz",
    "metal_nut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz",
    "pill": "https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz",
    "screw": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130-1629953152/screw.tar.xz",
    "tile": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz",
    "toothbrush": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz",
    "transistor": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166-1629953277/transistor.tar.xz",
    "wood": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938383-1629953354/wood.tar.xz",
    "zipper": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385-1629953449/zipper.tar.xz",
}


class MVTecDataset:
    def __init__(self, class_name: str, size: int = 224):
        self.class_name = class_name
        # you get a specific class
        self.size = size
        if class_name in MVTEC_CLASSES:
            # check if it is inside, please please download it
            self._download(MVTEC_CLASSES[class_name])
        # train dataset
        self.train_ds = MVTecTrainDataset(class_name, size)
        self.test_ds = MVTecTestDataset(class_name, size)

    def _download(self, url: str):
        if not isdir(DATASETS_PATH / self.class_name):
            print(
                f"   Could not find '{self.class_name}' in '{DATASETS_PATH}/'. Downloading ... "
            )
            wget.download(url)
            with tarfile.open(f"{self.class_name}.tar.xz") as tar:
                tar.extractall(DATASETS_PATH)
            os.remove(f"{self.class_name}.tar.xz")
            print("")  # force newline
        else:
            print(f"   Found '{self.class_name}' in '{DATASETS_PATH}/'\n")

    def get_datasets(self):
        return self.train_ds, self.test_ds

    def get_dataloaders(self):
        return DataLoader(self.train_ds), DataLoader(self.test_ds)


class MVTecTrainDataset(ImageFolder):
    # fakr imagefolder works in this way
    #         root/dog/xxx.png
    #         root/dog/xxy.png
    def __init__(self, class_name: str, size: int):
        super().__init__(
            root=DATASETS_PATH / class_name / "train",
            transform=transforms.Compose(
                [
                    transforms.Resize(
                        256, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            ),
        )
        self.class_name = class_name
        self.size = size
        # you always do resize then center crop

class MVTecTestDataset(ImageFolder):
    def __init__(self, class_name: str, size: int):
        super().__init__(
            root=DATASETS_PATH / class_name / "test",
            transform=transforms.Compose(
                [
                    transforms.Resize(
                        256, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            ),
            target_transform=transforms.Compose(
                [
                    transforms.Resize(
                        256, interpolation=transforms.InterpolationMode.NEAREST
                    ),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                ]
            ),
        )
        self.class_name = class_name
        self.size = size

    def __getitem__(self, index):
        path, _ = self.samples[index]
        # when you set loader it bring back to you that image
        sample = self.loader(path)

        if "good" in path:
            # the target is the gorund trueht heat map
            target = Image.new("L", (self.size, self.size))
            # loading the image grey scale
            # this is class 0
            # good class
            sample_class = 0
        else:
            target_path = path.replace("test", "ground_truth")
            target_path = target_path.replace(".png", "_mask.png")
            target = self.loader(target_path)
            # this is class 1
            sample_class = 1

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target[:1], sample_class


class StreamingDataset:
    """This dataset is made specifically for the streamlit app.
        we can ignore this dataset """

    def __init__(self, size: int = 224):
        self.size = size
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.samples = []

    def add_pil_image(self, image: Image):
        image = image.convert("RGB")
        self.samples.append(image)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return (self.transform(sample), tensor(0.0))
