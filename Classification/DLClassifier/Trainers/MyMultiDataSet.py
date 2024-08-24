from torch.utils.data import Dataset
import torch

def get_data_template(name):
    if name == "MyDataset":
        return MyMultiDataset

# class MyDataset(Dataset):
#
#     def __init__(self, Xs, y):
#         self.Xs = Xs
#         self.y = y
#
#     def __len__(self):
#         return len(self.y)
#
#     def __getitem__(self, index):
#         # 返回数据和对应的目标
#         return [X[index] for X in self.Xs], self.y[index]

class MyMultiDataset(Dataset):

    def __init__(self, Xs, y):
        self.Xs = Xs
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # 返回数据和对应的目标
        return [X[index] for X in self.Xs], self.y[index]

    @staticmethod
    def obtain_instance(kwargs):
        Xs, y = kwargs["Xs"], kwargs["y"]
        return MyMultiDataset(Xs=Xs, y=y)

    @staticmethod
    def collate_fn(batch):
        data = [item[0] for item in batch]
        data = [torch.stack([sublist[i] for sublist in data]) for i in range(len(data[0]))]
        labels = torch.stack([item[1] for item in batch])

        return data, labels
