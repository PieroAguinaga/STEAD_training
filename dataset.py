import torch.utils.data as data
import numpy as np
import torch
import random
torch.set_float32_matmul_precision('medium')
import option
import os

class Dataset(data.Dataset):
    def __init__(self, dataset_path, RGB_list , test_mode=False):

        if test_mode:
            self.rgb_list_file = RGB_list
        else:
            self.rgb_list_file = RGB_list

        self.test_mode = test_mode
        self.list = list(open(self.rgb_list_file))

        self.normals = [x for x in self.list if "Normal" in x]
        self.anomalies = [x for x in self.list if "Normal" not in x]
        self.n_len = len(self.normals)
        self.a_len = len(self.anomalies)
        self.dataset_path = dataset_path

    def __getitem__(self, index):
        if not self.test_mode:
            if index == 0:
                self.a_ind = [i for i, x in enumerate(self.list) if "Normal" not in x]
                self.n_ind = [i for i, x in enumerate(self.list) if "Normal"    in x]
                random.shuffle(self.a_ind)
                random.shuffle(self.n_ind)

            nindex = self.n_ind.pop()
            aindex = self.a_ind.pop()

            path = os.path.join(self.dataset_path, self.list[nindex].strip())
            nfeatures = np.load(path, allow_pickle=True)
            nfeatures = np.array(nfeatures, dtype=np.float32)
            nlabel = 0.0 if "Normal" in path else 1.0

            path = os.path.join(self.dataset_path, self.list[aindex].strip('\n'))
            afeatures = np.load(path, allow_pickle=True)
            afeatures = np.array(afeatures, dtype=np.float32)
            alabel = 0.0 if "Normal" in path else 1.0

            return nfeatures, nlabel, afeatures, alabel
    
        else:
            path = os.path.join(self.dataset_path, self.list[index].strip('\n'))
            features = np.load(path, allow_pickle=True)
            label = 0.0 if "Normal" in path else 1.0
            return features, label

    def __len__(self):

        if self.test_mode:
            return len(self.list)
        else:
            return min(self.a_len, self.n_len)
        

