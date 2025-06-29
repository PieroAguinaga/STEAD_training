import torch.utils.data as data
import numpy as np
import torch
import random
import os

class Dataset(data.Dataset):
    def __init__(self, dataset_path, RGB_list, test_mode=False, random_seed=2025):
        self.test_mode = test_mode
        self.dataset_path = dataset_path
        self.random_seed = random_seed

        # Leer y limpiar el archivo de lista, quitar extensión
        with open(RGB_list, 'r') as f:
            self.list = [os.path.splitext(line.strip())[0] for line in f if line.strip()]

        # Separar en normales y anómalos
        self.normals = [x for x in self.list if "Normal" in x]
        self.anomalies = [x for x in self.list if "Normal" not in x]

        self.n_len = len(self.normals)
        self.a_len = len(self.anomalies)

        # Inicializar estado aleatorio para sampling reproducible
        self._reset_indices()

    def _reset_indices(self):
        """Reinicia y remezcla los índices"""
        if not self.test_mode:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)

            self.normal_indices = random.sample(range(self.n_len), self.n_len)
            self.anomaly_indices = random.sample(range(self.a_len), self.a_len)

    def __getitem__(self, index):
        if not self.test_mode:
            # Rehacer shuffle si se acaban
            if len(self.normal_indices) == 0 or len(self.anomaly_indices) == 0:
                self._reset_indices()

            n_idx = self.normal_indices.pop()
            a_idx = self.anomaly_indices.pop()

            n_path = os.path.join(self.dataset_path, self.normals[n_idx] + ".npy")
            a_path = os.path.join(self.dataset_path, self.anomalies[a_idx] + ".npy")

            if not os.path.exists(n_path):
                raise FileNotFoundError(f"❌ Archivo no encontrado: {n_path}")
            if not os.path.exists(a_path):
                raise FileNotFoundError(f"❌ Archivo no encontrado: {a_path}")

            nfeatures = np.load(n_path, allow_pickle=True).astype(np.float32)
            afeatures = np.load(a_path, allow_pickle=True).astype(np.float32)

            return nfeatures, 0.0, afeatures, 1.0

        else:
            sample = self.list[index]
            path = os.path.join(self.dataset_path, sample + ".npy")
            if not os.path.exists(path):
                raise FileNotFoundError(f"❌ Archivo no encontrado: {path}")
            features = np.load(path, allow_pickle=True).astype(np.float32)
            label = 0.0 if "Normal" in sample else 1.0
            return features, label

    def __len__(self):
        return len(self.list) if self.test_mode else min(self.n_len, self.a_len)
