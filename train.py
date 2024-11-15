from src.dataloader import CityScapes
from src.model import UNET
from src.trainer import train
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

DATA_PATH = "data"
TRAIN_TEST_SPLIT = [0.8, 0.2]


if __name__ == "__main__":
    transforms = A.Compose([
        A.Resize(256,256),
        A.Normalize(),
        ToTensorV2()
    ])
    data = CityScapes(os.path.join(DATA_PATH, "CameraRGB"), os.path.join(DATA_PATH, "CameraSeg"), transforms=transforms)
    #how many classes inside the data
    all_class = []
    for i in range(len(data)):
        all_class.extend(data[i][1].unique().tolist())

    all_class = list(set(all_class))

    print(all_class)
    print(len(data))
    model = UNET(
        in_channels=3,
        out_channels=len(all_class)
    )
    
    train_datasets, test_datasets = torch.utils.data.random_split(data, TRAIN_TEST_SPLIT)
    train(
        model=model,
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        num_epochs=200
    )