import torch
from torch.utils.data.dataset import Dataset


class gifDataset(Dataset):
    def __init__(self, X, y, of) -> None:
        super().__init__()
        self.X = X
        self.y = y
        self.of = of

    def __getitem__(self, index):
        if self.of is not None:
            return (self.X[index, ...], self.y[index, ...], self.of[index, ...])
        else:
            return (self.X[index, ...], self.y[index, ...])

    def __len__(self):
        return self.X.shape[0]
