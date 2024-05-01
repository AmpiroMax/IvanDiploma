from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch
import os


class ElectroInterDataset(Dataset):
    def __init__(self, data_foulder_path: str) -> None:
        self.data = dict()

        # ====================================================================
        # Reading data from files
        # ====================================================================
        for i, file_name in enumerate(os.listdir(data_foulder_path+"/images")):
            image = Image.open(data_foulder_path+"/images/"+file_name)
            label = Image.open(data_foulder_path+"/labels/"+file_name)
            scale_file_name = file_name.split(".")[0]+".txt"
            scale = self._open_scale(
                data_foulder_path+"/scales/"+scale_file_name)

            self.data[i] = (image, label, scale)

        # ====================================================================
        # Preparing transforms
        # ====================================================================
        self.transform = transforms.Compose([
            transforms.PILToTensor()
        ])

    def get_sample(self, idx) -> tuple:
        return self.data[idx]

    def _open_scale(self, path: str) -> dict:
        with open(path) as f:
            lines = f.readlines()
        scale = {
            i: float(value.strip()) for i, value in enumerate(lines)
        }
        return scale

    def __getitem__(self, idx):
        image_tensor = self.transform(self.data[idx][0]).permute(2, 0, 1)[
            None, ...].to("cuda")
        label_tensor = self.transform(self.data[idx][1]).permute(2, 0, 1)[
            None, ...].to("cuda")
        item = (image_tensor, label_tensor)

        return item

    def __len__(self):
        return len(self.data)
