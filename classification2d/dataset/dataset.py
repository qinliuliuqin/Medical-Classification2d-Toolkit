import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def read_label_file(image_list_file, mode):
  """
  Reads the training image list file and returns a list of image file names.
  """
  images_df = pd.read_csv(image_list_file)
  image_name_list = images_df['image_name'].tolist()
  label_list = None

  if mode == 'train' or mode == 'val':
    label_list = images_df['label'].tolist()

  return image_name_list, label_list


class ClassificationDataset(Dataset):
    """ training data set for volumetric segmentation """

    def __init__(self, mode, data_folder, label_file, transforms=None):
        if label_file.endswith('csv'):
            self.image_name_list, self.label_list = read_label_file(label_file, mode)
        else:
            raise ValueError('label file must be a csv file')

        self.mode = mode
        self.data_folder = data_folder
        self.transforms = transforms

    def __len__(self):
        """ get the number of images in this data set """
        return len(self.image_name_list)

    def __getitem__(self, idx):
        """ get a training sample - image(s) and segmentation pair
        :param idx:  the sample index
        :return cropped image, cropped mask, crop frame, case name
        """
        image_name = self.image_name_list[idx]

        # load image
        img_path = os.path.join(self.data_folder, image_name)
        img = Image.open(img_path).convert("RGB")

        if self.mode == 'train' or self.mode == 'val':
            label = self.label_list[idx]
            if self.transforms is not None:
                img = self.transforms(img)

        elif self.mode == 'test':
            label = None
            if self.transforms is not None:
                img = self.transforms(img)
        else:
            raise ValueError('Unsupported mode.')

        return img, label