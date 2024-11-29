import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.nn import functional as F

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, label_mapping=None):
        self.X = X
        self.y = y
        self.label_mapping = label_mapping or self.create_label_mapping(y)
        self.y_mapped = [self.label_mapping[label] for label in y]


    def create_label_mapping(self, y):
        unique_labels = set(y)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        return label_mapping

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.Tensor(self.X[index]), torch.LongTensor([self.y_mapped[index]]).squeeze()