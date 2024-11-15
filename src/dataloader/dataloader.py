import torch
import cv2
import os

class CityScapes(torch.utils.data.DataLoader):
  def __init__(self, image_folder_path, mask_folder_path, transforms=None):
    self.images_folder = image_folder_path
    self.masks_folder = mask_folder_path
    self.images = os.listdir(self.images_folder)

    self.transforms = transforms

  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
    file_name = self.images[index]
    image = cv2.imread(os.path.join(self.images_folder, file_name))
    mask = cv2.imread(os.path.join(self.masks_folder, file_name))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = mask[:,:,2]

    if self.transforms is not None:
      augmented = self.transforms(image=image, mask=mask)
      image = augmented['image']
      mask = augmented['mask']


    return image, mask.unsqueeze(0)
