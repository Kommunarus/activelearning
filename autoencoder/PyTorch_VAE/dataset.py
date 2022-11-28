import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
from PIL import Image
import numpy as np
from scripts.classification.unit import prepare_items, read_dirs_dataset



# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True


class Dataset_flowers(Dataset):
    def __init__(self, data_dir, split, transform, filter_label, limit):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        path_to_dataset_train = os.path.join(data_dir, 'train')
        dict_id, num_labels = read_dirs_dataset(path_to_dataset_train)
        # all_items = [x for x in dict_id.keys()]
        if not filter_label is None:
            del_labels = [filter_label, ]
        else:
            del_labels = []

        if split == 'train':
            labeled_data = sorted(prepare_items(limit=limit,
                                                train=True, files=dict_id, test_size=0.2,
                                                del_labels=del_labels))
        elif split == 'test':
            labeled_data = sorted(prepare_items(val=True, files=dict_id, test_size=0.2,
                                                del_labels=del_labels))

        files = []
        # if not filter_label is None:
        #     if filter_label == 0:
        #         path2 = os.path.join(path_to_dataset_train, '1')
        #     if filter_label == 1:
        #         path2 = os.path.join(path_to_dataset_train, '0')
        #     #     files_in_dir = os.listdir(path2)
        #     files = files + [os.path.join(path2, x) for x in labeled_data]
        # else:
        # label_all = os.listdir(path_to_dataset_train)
        # for label in label_all:
        files = files + [os.path.join(path_to_dataset_train, dict_id[x][0], x) for x in labeled_data]

        self.data = files
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file = self.data[idx]
        image = Image.open(file)
        image = image.resize((224, 224))

        if len(image.size) == 2:
            image = image.convert('RGB')
        # np_img = np.array(image)
        if self.transform:
            image = self.transform(image)
        return image, idx


class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        filter_label=None,
        limit=-1,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.filter_label = filter_label
        self.limit = limit

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose([
                                              transforms.RandomHorizontalFlip(),
                                              # transforms.CenterCrop(148),
                                              transforms.Resize(self.patch_size),
                                              transforms.ToTensor(),])
        
        val_transforms = transforms.Compose([
                                            # transforms.RandomHorizontalFlip(),
                                            # transforms.CenterCrop(148),
                                            transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),])
        
        self.train_dataset = Dataset_flowers(self.data_dir, split='train', transform=train_transforms,
                                             filter_label=self.filter_label, limit=self.limit)
        self.val_dataset = Dataset_flowers(self.data_dir, split='test', transform=val_transforms,
                                           filter_label=self.filter_label, limit=-1)
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     