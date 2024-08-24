import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


from Classification.DLClassifier.Models.RootModel import ModelWrapper


def get_dl_model(config, layer):
    model_type = config.get("ModelType", None)
    assert model_type != None, "深度学习中的模型配置信息不能为空"
    if model_type == "DNN":
        return DNNWrapper(model_type, config, layer)
    elif model_type == "BNN":
        return BNNWrapper(model_type, config, layer)
    elif model_type == "GateBNN":
        return GateBNNWrapper(model_type, config, layer)
    elif model_type == "GNN":
        return GNNWrapper(model_type, config, layer)
    elif model_type == "TwoDNN":
        return TwoDNNWrapper(model_type, config, layer)
    elif model_type == "TwoBNN":
        return TwoBNNWrapper(model_type, config, layer)
    elif model_type == "TwoGateBNN":
        return  TwoGateBNNWrapper(model_type, config, layer)
    elif model_type == "ThreeDNN":
        return ThreeDNNWrapper(model_type, config, layer)
    elif model_type == "ThreeBNN":
        return ThreeBNNWrapper(model_type, config, layer)
    elif model_type == "ThreeGateBNN":
        return ThreeGateBNNWrapper(model_type, config, layer)
    else:
        raise ""

class DNNWrapper(ModelWrapper):

    def __init__(self, model_type, configs, layer):
        super(DNNWrapper, self).__init__(model_type, DNNWrapper.DNN, configs, layer)

    class DNN(nn.Module):
        def __init__(self, InputSize, ClassNum, HiddenParameter):
            super(DNNWrapper.DNN, self).__init__()
            assert len(InputSize) == 1, self.__class__.__name__ + "只支持输入模态为 1"
            self.branch = nn.Sequential(nn.Linear(InputSize[0], HiddenParameter, bias=False), nn.ReLU(), nn.BatchNorm1d(HiddenParameter))
            self.dense1 = nn.Linear(HiddenParameter, ClassNum)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            feature = self.branch(x)
            out = self.dense2(feature)
            out = self.softmax(out)
            return feature, out

class BNNWrapper(ModelWrapper):

    def __init__(self, model_type,  configs, layer):
        super(BNNWrapper, self).__init__(model_type,  BNNWrapper.BNN, configs, layer)

    class BNN(nn.Module):
        def __init__(self, InputSize, ClassNum, HiddenParameter):
            super(BNNWrapper.BNN, self).__init__()
            assert len(InputSize) == 1, self.__class__.__name__ + "只支持输入模态为 1"
            self.branch_1 = nn.Sequential(nn.Linear(InputSize[0], HiddenParameter, bias=False), nn.ReLU(), nn.BatchNorm1d(HiddenParameter))
            self.branch_2 = nn.Sequential(nn.Linear(InputSize[0], HiddenParameter, bias=False), nn.ReLU(), nn.BatchNorm1d(HiddenParameter))
            self.dense2 = nn.Linear(HiddenParameter, ClassNum)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            f1 = self.branch_1(x)
            f2 = self.branch_2(x)
            feature = torch.mul(f1, f2)

            out = self.dense2(feature)
            out = self.softmax(out)

            return feature, out


class GateBNNWrapper(ModelWrapper):

    def __init__(self, model_type,  configs, layer):
        super(GateBNNWrapper, self).__init__(model_type,  GateBNNWrapper.GateBNN, configs, layer)

    class GateBNN(nn.Module):
        def __init__(self, InputSize, ClassNum, HiddenParameter):
            super(GateBNNWrapper.GateBNN, self).__init__()
            assert len(InputSize) == 1, self.__class__.__name__ + "只支持输入模态为 1"
            self.branch_1 = nn.Sequential(nn.Linear(InputSize[0], HiddenParameter, bias=False), nn.ReLU(), nn.BatchNorm1d(HiddenParameter))
            self.branch_2 = nn.Sequential(nn.Linear(InputSize[0], HiddenParameter, bias=False), nn.ReLU(), nn.BatchNorm1d(HiddenParameter))
            self.sig_1 = nn.Sequential(nn.Linear(InputSize[0], HiddenParameter, bias=False), nn.Sigmoid())
            self.dense2 = nn.Linear(HiddenParameter, ClassNum)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            f1 = self.branch_1(x)
            f1_sig = self.sig_1(x)
            f1 = torch.mul(f1, f1_sig)

            f2 = self.branch_2(x)
            feature = torch.mul(f1, f2)

            out = self.dense2(feature)
            out = self.softmax(out)

            return feature, out

class GNNWrapper(ModelWrapper):

    def __init__(self, model_type,  configs, layer):
        super(GNNWrapper, self).__init__(model_type,  GNNWrapper.GNN, configs, layer)

    class GNN(nn.Module):
        def __init__(self, InputSize, ClassNum, HiddenParameter, DistanceType="Euclidean"):
            assert len(InputSize) == 1, self.__class__.__name__ + "只支持输入模态为 1"
            super(GNNWrapper.GNN, self).__init__()
            self.distance_type = DistanceType
            self.conv1 = nn.Sequential(nn.Linear(InputSize[0], InputSize[0], bias=False),
                                          nn.Tanh(),
                                          nn.BatchNorm1d(InputSize[0]),
                                          nn.Linear(InputSize[0], HiddenParameter, bias=False))
            self.conv2 = nn.Sequential(nn.Linear(HiddenParameter, HiddenParameter, bias=False),
                                          nn.Tanh(),
                                          nn.BatchNorm1d(HiddenParameter),
                                          nn.Linear(HiddenParameter, ClassNum, bias=False))

        def forward(self, x):
            distance_type = self.obtain_distance_type()
            dis_matrix = self.obtain_distance_matrix(x, distance_type)
            x = torch.mm(dis_matrix, x)
            feat = self.conv1(x)
            dis_matrix = self.obtain_distance_matrix(feat, distance_type)
            out = torch.mm(dis_matrix, feat)
            out = self.conv2(out)
            return feat, out

        def obtain_distance_matrix(self, feat, distance_type):
            if distance_type == "Euclidean":
                dis_matrix = torch.cdist(feat, feat, p=2)
            return dis_matrix

        def obtain_distance_type(self):
            return self.distance_type

class TwoDNNWrapper(ModelWrapper):

    def __init__(self, model_type,  configs, layer):
        super(TwoDNNWrapper, self).__init__(model_type,  TwoDNNWrapper.DNN, configs, layer)

    class DNN(nn.Module):

        def __init__(self, InputSize, ClassNum, HiddenParameter):
            super(TwoDNNWrapper.DNN, self).__init__()
            assert len(InputSize) == 2, self.__class__.__name__ + "只支持输入模态为 2"
            self.branch_1 = nn.Sequential(nn.Linear(InputSize[0], HiddenParameter, bias=False), nn.ReLU(), nn.BatchNorm1d(HiddenParameter))
            self.branch_2 = nn.Sequential(nn.Linear(InputSize[1], HiddenParameter, bias=False), nn.ReLU(), nn.BatchNorm1d(HiddenParameter))

            self.dense2 = nn.Linear(HiddenParameter * 2, ClassNum)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x1, x2):
            f1 = self.branch_1(x1)
            f2 = self.branch_2(x2)

            feature = torch.cat([f1, f2], axis=1)

            out = self.dense2(feature)
            out = self.softmax(out)
            return feature, out

class TwoBNNWrapper(ModelWrapper):

    def __init__(self, model_type,  configs, layer):
        super(TwoBNNWrapper, self).__init__(model_type,  TwoBNNWrapper.BNN, configs, layer)

    class BNN(nn.Module):

        def __init__(self, InputSize, ClassNum, HiddenParameter):
            super(TwoBNNWrapper.BNN, self).__init__()
            assert len(InputSize) == 2, self.__class__.__name__ + "只支持输入模态为 2"
            self.branch_1 = nn.Sequential(nn.Linear(InputSize[0], HiddenParameter, bias=False), nn.ReLU(), nn.BatchNorm1d(HiddenParameter))
            self.branch_2 = nn.Sequential(nn.Linear(InputSize[1], HiddenParameter, bias=False), nn.ReLU(), nn.BatchNorm1d(HiddenParameter))

            self.dense2 = nn.Linear(HiddenParameter, ClassNum)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x1, x2):
            f1 = self.branch_1(x1)
            f2 = self.branch_2(x2)

            feature = torch.mul(f1, f2)

            out = self.dense2(feature)
            out = self.softmax(out)
            return feature, out

class TwoGateBNNWrapper(ModelWrapper):

    def __init__(self, model_type,  configs, layer):
        super(TwoGateBNNWrapper, self).__init__(model_type,  TwoGateBNNWrapper.BNN, configs, layer)

    class BNN(nn.Module):

        def __init__(self, InputSize, ClassNum, HiddenParameter):
            super(TwoGateBNNWrapper.BNN, self).__init__()
            assert len(InputSize) == 2, self.__class__.__name__ + "只支持输入模态为 2"
            self.branch_1 = nn.Sequential(nn.Linear(InputSize[0], HiddenParameter, bias=False), nn.ReLU(), nn.BatchNorm1d(HiddenParameter))
            self.branch_2 = nn.Sequential(nn.Linear(InputSize[1], HiddenParameter, bias=False), nn.ReLU(), nn.BatchNorm1d(HiddenParameter))

            self.sig_1 = nn.Sequential(nn.Linear(InputSize[0], HiddenParameter, bias=False), nn.Sigmoid())
            self.sig_2 = nn.Sequential(nn.Linear(InputSize[1], HiddenParameter, bias=False), nn.Sigmoid())

            self.dense2 = nn.Linear(HiddenParameter, ClassNum)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x1, x2):
            f1 = self.branch_1(x1)
            f1_sig = self.sig_1(x1)
            f1 = torch.mul(f1, f1_sig)

            f2 = self.branch_2(x2)
            f2_sig = self.sig_2(x2)
            f2 = torch.mul(f2, f2_sig)

            feature = torch.mul(f1, f2)

            out = self.dense2(feature)
            out = self.softmax(out)
            return feature, out

class ThreeDNNWrapper(ModelWrapper):
    def __init__(self, model_type,  configs, layer):
        super(ThreeDNNWrapper, self).__init__(model_type,  ThreeDNNWrapper.BNN, configs, layer)

    class DNN(nn.Module):

        def __init__(self, InputSize, ClassNum, HiddenParameter):
            super(ThreeDNNWrapper.DNN, self).__init__()
            assert len(InputSize) == 3, self.__class__.__name__ + "只支持输入模态为 3"
            self.branch_1 = nn.Sequential(nn.Linear(InputSize[0], HiddenParameter, bias=False), nn.ReLU(), nn.BatchNorm1d(HiddenParameter))
            self.branch_2 = nn.Sequential(nn.Linear(InputSize[1], HiddenParameter, bias=False), nn.ReLU(), nn.BatchNorm1d(HiddenParameter))
            self.branch_3 = nn.Sequential(nn.Linear(InputSize[2], HiddenParameter, bias=False), nn.ReLU(), nn.BatchNorm1d(HiddenParameter))

            self.dense2 = nn.Linear(32 * 3, ClassNum)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x1, x2, x3):
            f1 = self.branch_1(x1)
            f2 = self.branch_2(x2)
            f3 = self.branch_3(x3)

            feat = torch.cat([f1, f2, f3], axis=1)

            out = self.dense2(feat)
            out = self.softmax(out)
            return feat, out


class ThreeBNNWrapper(ModelWrapper):
    def __init__(self, model_type,  configs, layer):
        super(ThreeBNNWrapper, self).__init__(model_type,  ThreeBNNWrapper.BNN, configs, layer)

    class BNN(nn.Module):

        def __init__(self, InputSize, ClassNum, HiddenParameter):
            super(ThreeBNNWrapper.BNN, self).__init__()
            assert len(InputSize) == 3, self.__class__.__name__ + "只支持输入模态为 3"
            self.branch_1 = nn.Sequential(nn.Linear(InputSize[0], HiddenParameter, bias=False), nn.ReLU(), nn.BatchNorm1d(HiddenParameter))
            self.branch_2 = nn.Sequential(nn.Linear(InputSize[1], HiddenParameter, bias=False), nn.ReLU(), nn.BatchNorm1d(HiddenParameter))
            self.branch_3 = nn.Sequential(nn.Linear(InputSize[2], HiddenParameter, bias=False), nn.ReLU(), nn.BatchNorm1d(HiddenParameter))

            self.dense2 = nn.Linear(32, ClassNum)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x1, x2, x3):
            f1 = self.branch_1(x1)
            f2 = self.branch_2(x2)
            f3 = self.branch_3(x3)

            feat = torch.mul(f1, f2)
            feat = torch.mul(feat, f3)

            out = self.dense2(feat)
            out = self.softmax(out)
            return feat, out

class ThreeGateBNNWrapper(ModelWrapper):
    def __init__(self, model_type, configs, layer):
        super(ThreeGateBNNWrapper, self).__init__(model_type, ThreeGateBNNWrapper.BNN, configs, layer)

    class BNN(nn.Module):

        def __init__(self, InputSize, ClassNum, HiddenParameter):
            super(ThreeGateBNNWrapper.BNN, self).__init__()
            assert len(InputSize) == 3, self.__class__.__name__ + "只支持输入模态为 3"
            self.branch_1 = nn.Sequential(nn.Linear(InputSize[0], HiddenParameter, bias=False), nn.ReLU(), nn.BatchNorm1d(HiddenParameter))
            self.branch_2 = nn.Sequential(nn.Linear(InputSize[1], HiddenParameter, bias=False), nn.ReLU(), nn.BatchNorm1d(HiddenParameter))
            self.branch_3 = nn.Sequential(nn.Linear(InputSize[2], HiddenParameter, bias=False), nn.ReLU(), nn.BatchNorm1d(HiddenParameter))

            self.sig_1 = nn.Sequential(nn.Linear(InputSize[0], HiddenParameter, bias=False), nn.Sigmoid())
            self.sig_2 = nn.Sequential(nn.Linear(InputSize[1], HiddenParameter, bias=False), nn.Sigmoid())
            self.sig_3 = nn.Sequential(nn.Linear(InputSize[2], HiddenParameter, bias=False), nn.Sigmoid())

            self.dense2 = nn.Linear(HiddenParameter, ClassNum)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x1, x2, x3):
            f1 = self.branch_1(x1)
            f1_sig = self.sig_1(x1)
            f1 = torch.mul(f1, f1_sig)

            f2 = self.branch_2(x2)
            f2_sig = self.sig_2(x2)
            f2 = torch.mul(f2, f2_sig)

            f3 = self.branch_3(x3)
            f3_sig = self.sig_3(x3)
            f3 = torch.mul(f3, f3_sig)

            feat = torch.mul(f1, f2)
            feat = torch.mul(feat, f3)

            out = self.dense2(feat)
            out = self.softmax(out)
            return feat, out