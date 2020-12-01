from torch.utils.data import Dataset
from abc import abstractmethod
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from os import path
import cv2

from utils import LoggingParent


class BaseDataset(Dataset,LoggingParent):

    def __init__(self, config, transforms, train=True, datakeys = None):
        Dataset.__init__(self)
        LoggingParent.__init__(self)
        if datakeys is None:
            self.datakeys = ["images"]
        if not path.isdir(config.datapath):
            self.basepath = None
        else:
            self.basepath = config.datapath

        self.spatial_size = config.reconstr_dim
        self.transforms = transforms
        self.train = train

        self.datadict = {"img_path":[]}
        self._read_data()
        self.datadict = {key:np.asarray(self.datadict[key]) for key in self.datadict}

        assert self.basepath is not None
        assert self.datadict["img_path"].shape[0] > 0

        self._output_dict = {"images": self._get_img}

        self.logger.info(f'Constructed {self.__class__.__name__} in {"train" if self.train else "test"}-mode; dataset consists of {self.__len__()} samples.')



    def __len__(self):
        return self.datadict["img_path"].shape[0]


    def __getitem__(self, idx):
        data = {key: self._output_dict[key](idx) for key in self.datakeys}
        return data

    @abstractmethod
    def _read_data(self):
        pass


    def _get_img(self,idx):

        img_path = self.datadict["img_path"][idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(self.spatial_size,self.spatial_size))

        img = self.transforms(img)

        return img






class DeepFashionDataset(BaseDataset):

    def _read_data(self,):
        metafile = "/export/scratch/compvis/datasets/compvis-datasets/deepfashion_allJointsVisible/data.csv"
        ddf = pd.read_csv(metafile)

        img_paths = ddf["filename"]
        train_groups, test_groups = train_test_split(img_paths, test_size=0.2)

        target_paths = train_groups if self.train else test_groups

        if self.basepath is None:
            self.basepath = "/export/scratch/compvis/datasets/deepfashion_inshop/Img/img/"

        self.datadict["img_path"] = [path.join(self.basepath,p) for p in target_paths]




__datasets__ = {"DeepFashion": DeepFashionDataset}

def get_dataset(dataset_name):
    return __datasets__[dataset_name]


if __name__ == '__main__':
    pass