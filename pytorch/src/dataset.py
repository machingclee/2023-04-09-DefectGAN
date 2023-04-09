import os
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, image
from torchvision import transforms as T
from device import device
from config import ModelAndTrainingConfig as config


class DefectDataset(Dataset):
    def __init__(self):
        super(DefectDataset, self).__init__()
        self.defect_filepath_arr = None
        self.load_fect_filepath()

    def load_fect_filepath(self):
        if self.defect_filepath_arr is None:
            defect_data = []
            dataset_dir = config.dataset_dir

            for defect in config.defect_labels:
                normal_dir = f"{dataset_dir}/{defect}/normal"
                defect_dir = f"{dataset_dir}/{defect}/defect"
                mask_dir = f"{dataset_dir}/{defect}/defect_segmentation"
                defect_index = config.labels.index(defect)

                for basename in os.listdir(normal_dir):
                    filename = basename.replace(".jpg", "")
                    normal_path = f"{normal_dir}/{filename}.jpg"
                    defect_path = f"{defect_dir}/{filename}.jpg"
                    defect_mask_path = f"{mask_dir}/{filename}.png"
                    defect_data.append([defect_index, normal_path, defect_path, defect_mask_path])

            self.defect_filepath_arr = defect_data

    def load_cls_index_and_imgs_from_index(self, index):
        cls_index, normal_path, defect_path, defect_seg_path = self.defect_filepath_arr[index]
        resize = T.Resize(config.image_shape[1:3])
        np_normal = resize(read_image(normal_path)) / 127.5 - 1
        np_defect = resize(read_image(defect_path)) / 127.5 - 1
        np_defect_mask = resize(read_image(defect_seg_path, mode=image.ImageReadMode.GRAY)) / 255
        np_defect_mask = torch.where(np_defect_mask > 0.5, 1, 0)
        return cls_index, np_normal, np_defect, np_defect_mask

    def load_spatial_charactergorical_map_from_index(self, index):
        cls_index, _, _, defect_mask = self.load_cls_index_and_imgs_from_index(index)
        spartial_dim = tuple(config.image_shape[1:3])
        spatial_cat_map = np.zeros((len(config.labels),) + spartial_dim)
        spatial_cat_map[cls_index] = defect_mask[0]
        return spatial_cat_map

    def get_num_of_batches(self):
        return (len(self.defect_filepath_arr) // config.batch_size) + (0 if config.dataset_drop_last else 1)

    def __getitem__(self, index):
        defect_cls_index, np_normal, np_defect, np_defect_seg = \
            self.load_cls_index_and_imgs_from_index(index)

        spatial_cat_map = self.load_spatial_charactergorical_map_from_index(index)

        return (
            torch.as_tensor(defect_cls_index).type(torch.LongTensor).to(device),
            torch.as_tensor(np_normal, dtype=torch.float, device=device),
            torch.as_tensor(np_defect, dtype=torch.float, device=device),
            torch.as_tensor(np_defect_seg, dtype=torch.float, device=device),
            torch.as_tensor(spatial_cat_map, dtype=torch.float, device=device)
        )

    def __len__(self):
        return len(self.defect_filepath_arr)


defect_dataset = DefectDataset()
defectDataloader = DataLoader(dataset=defect_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              #   prefetch_factor=config.batch_size,
                              drop_last=config.dataset_drop_last,
                              num_workers=config.dataset_num_workers)

if __name__ == "__main__":
    print(defectDataloader.dataset[0])
