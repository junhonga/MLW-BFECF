import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import copy

from Classification.DLClassifier.Models.BaseModel import get_dl_model
from Classification.DLClassifier.Trainers.BaseTrainer import get_dl_trainer
from Classification.MLClassifier.BaseClassify import get_ml_base_classifier
from Classification.MLClassifier.EnsembleClassify import get_ens_classifier
from ClassifierBuider.PreProcessors.ProcessorsDispatcher import MultiClassifierBuiderProcessorsDispatcher
import time


def get_classifier_buider(builder_name, builder_type, modality_names, config):
    if builder_type == "ML":
        return MLClassifierBuider(builder_name, builder_type, modality_names, config)
    elif builder_type == "DL":
        return DLClassifierBuider(builder_name, builder_type, modality_names, config)
    else:
        raise "暂时不支持" + builder_name + "特征提取器/分类器的构建器"

class BuilderBuiderWrapper():

    def __init__(self, builder_name, builder_type, m_names, config):

        self.builder_name = builder_name
        self.builder_type = builder_type
        self.executable = True
        self.m_names = m_names
        self.debug = config.get("DeBug", True)

        self.classifier_type = config.get("ClassifierType", None)
        self.layer_scopes = config.get("LayerScopes", None)

        self.init_pre_processors(config)

    def init_pre_processors(self, config):
        processor_dispatcher = MultiClassifierBuiderProcessorsDispatcher()
        processor_configs = config.get("Processors", None)
        if processor_configs != None:
            self.processors = processor_dispatcher.obtain_instance(processor_configs)
        else:
            self.processors = []

    def execute_pre_processors(self, data, buider, cls_cfig, layer):
        for pre_processor in self.processors:
            if pre_processor.pre_executable(layer):
                data = pre_processor.pre_fit_execute(data, buider, layer)
        return data

    def execute_post_processors(self, data, buider, ests, cls_cfig, layer):
        for pre_processor in self.processors:
            if pre_processor.post_executable(layer):
                data, ests = pre_processor.post_fit_execute(data, buider, ests, layer)
        return data, ests

    def obtain_fit_classifier(self, data, layer=None):
        # 获得分类器的配置信息
        cls_cfig = self.obtain_classifier_cfig()
        # 最后一次检查需不需要进行训练, 注意这种检查不会影响到 self.fit_executable 的改变 (即之后级联层不会受到影响)
        fit_executable = self.check_classifier_init(cls_cfig, layer)
        if fit_executable:

            # 数据预处理
            data = self.execute_pre_processors(data, self, cls_cfig, layer)
            ests = self._obtain_fit_classifier(data, cls_cfig, layer)
            data, ests = self.execute_post_processors(data, self, ests, cls_cfig, layer)

            return ests
        else:
            ests, train_finfo = None, None
            return ests

    def add_infos_to_classifier_cfig(self):
        self.classifier_cfig["BuilderType"] = self.obtain_builder_type()
        self.classifier_cfig["ModalityNames"] = self.obtain_modality_name()

    def obtain_classifier_cfig(self):
        pass


    def obtain_classifier(self):
        pass

    def obtain_finfos(self, data, layer):
        # 封装一些相关信息
        train_finfo = data["TrainFinfo"]
        builder_type = self.obtain_builder_type()
        data_type = self.obtain_data_type()
        classifier_type = self.obtain_classifier_type()
        # 将一些信息附加到传回来的信息，以方便后面层的使用
        train_finfo = dict(BuilderType=builder_type, DataType=data_type, TrainFinfo=train_finfo,
                           ClassifierType=classifier_type, Layer=layer)
        return train_finfo

    def _obtain_fit_classifier(self, data, config, layer):
        pass

    def fit_executable(self, layer):
        return True

    def update_config(self, new_cfig, layer):
        pass

    def set_builder_name(self, builder_name):
        self.builder_name = builder_name

    def obtain_builder_name(self):
        return self.builder_name

    def set_builder_type(self, builder_type):
        self.builder_type = builder_type

    def obtain_builder_type(self):
        return self.builder_type

    def obtain_modality_name(self):
        return self.m_names

    def set_modality_name(self, m_names):
        self.m_names = m_names

    def obtain_executable(self):
        return self.executable

    def set_executable(self, executable):
        self.executable = executable

    def obtain_processors(self):
        return self.processors

    def set_processors(self, processors):
        self.processors = processors

    def obtain_classifier_configs(self):
        return self.classifier_configs

    def set_classifier_configs(self, new_classifier_configs):
        self.classifier_configs = new_classifier_configs

    def obtain_classifier_type(self):
        return self.classifier_type

    def set_layer_scopes(self, layer_scopes):
        self.layer_scopes = layer_scopes

    def obtain_layer_scopes(self):
        return self.layer_scopes

    def check_classifier_init(self, config, layer):
        # 这个方法是用于判断是否当前层需要初始化哪些基分类器
        layer_scopes = self.obtain_layer_scopes()
        # 如果当前配置没有设置 LayerScopes 参数, 直接返回 True
        if layer_scopes is None:
            return True
        # 检测当前层是不是 LayerScopes 参数中
        if layer in layer_scopes:
            return True
        return False


class MLClassifierBuider(BuilderBuiderWrapper):

    def __init__(self, builder_name, builder_type, m_names, config):
        super(MLClassifierBuider, self).__init__(builder_name, builder_type, m_names, config)

        self.classifier_cfig = config.get("ClassifierConfig", None)
        self.add_infos_to_classifier_cfig()


    def _obtain_fit_classifier(self, data, config, layer):
        # 给配置追加一些相关信息
        config["Name"] = config.get("ClassifierName")
        classifier_type = config.get("ClassifierType", None)
        config["ClassifierType"] = classifier_type

        # 获得分类器
        est = self.obtain_classifier(classifier_type, config, layer)
        # 训练一些分类器
        est = self._fit(data, est, layer)

        return est

    def _fit(self, data, est, layer):
        # 获得对应的模态
        m_names = self.obtain_modality_name()
        # 获得模态数据
        modality_data = self.obtain_modality_data(data, m_names)
        # 封装
        est.execute_fit_step(modality_data, layer)
        return est

    def obtain_modality_data(self, data, m_names):

        new_data = copy.deepcopy(data)
        # 获得训练集
        Xs_train, y_train = data["X_train"], data["y_train"]
        Xs_val, y_val = data.get("X_val", None), data.get("y_val", None)
        # 获得对应模态的训练集
        Xs_train = [Xs_train[m_name] for m_name in m_names]
        Xs_val = [Xs_val[m_name] for m_name in m_names]
        # 获得模态的数量
        m_names_len = len(m_names)
        # 根据模态进行数据处理
        if m_names_len == 1:
            Xs_train, Xs_val = Xs_train[0], Xs_val[0]
        elif m_names_len > 1 :
            Xs_train = np.concatenate(Xs_train, axis=1)
            Xs_val = np.concatenate(Xs_val, axis=1)
        # 封装数据
        new_data["X_train"] = Xs_train
        new_data["X_val"] = Xs_val
        return new_data

    def generate_name(self):
        est_name = self.config.get("Name", None)
        if est_name == None:
            est_name = self.est_type
        modality_names = [str(m_name) for m_name in self.modality_names]
        return est_name + "&" + "_".join(modality_names)

    def obtain_classifier(self, est_type, configs, layer,):
        est = get_ens_classifier(est_type, configs, layer, default=True) \
            or get_ml_base_classifier(est_type, configs, layer, default=True)
        if est == None:
            raise "暂时不支持" + est_type + "分类器/特征提取器"
        return est

    def obtain_classifier_cfig(self):
        return self.classifier_cfig

    def set_classifier_cfig(self, new_classifier_cfig):
        self.classifier_cfig = new_classifier_cfig
        self.classifier_cfig_num = len(new_classifier_cfig)

class DLClassifierBuider(BuilderBuiderWrapper):

    def __init__(self, builder_name, builder_type, modality_names, configs):
        super(DLClassifierBuider, self).__init__(builder_name, builder_type, modality_names, configs)

        configs = copy.deepcopy(configs)

        self.cuda = torch.cuda.is_available()
        self.trainer_cfig = configs.get("Trainer", None)
        assert self.trainer_cfig != None, "使用深度学习方法时，必须设置训练器"

        self.classifier_cfig = configs.get("Model", None)
        assert self.classifier_cfig != None, "使用深度学习方法时，必须设置模型"

        self.classifier_cfig_num = len(self.classifier_cfig)
        self.add_infos_to_classifier_cfig()

        self.loss_fun_cfig = configs.get("LossFun", None)
        assert self.loss_fun_cfig != None, "使用深度学习方法时，必须设置损失函数"

        self.optimizer_cfig = configs.get("Optimizer", None)
        assert self.optimizer_cfig != None, "使用深度学习方法时，必须设置优化器"

        self.debug = configs.get("DeBug", True)


    def _obtain_fit_classifier(self, data, model_config, layer):

        trainer_cfig = self.obtain_trainer_cfig()
        trainer_cfig = self.update_trainer_config(data, trainer_cfig, layer)
        trainer = self.obtain_trainer(trainer_cfig, layer)

        loss_fun_cfig = self.obtain_loss_fun_cfig()
        loss_fun_cfig = self.update_loss_fun_config(data, loss_fun_cfig, layer)
        loss_fun = self.obtain_loss_fun(loss_fun_cfig, layer)
        trainer.set_loss_fun(loss_fun)

        classifier_cfig = self.obtain_classifier_cfig()
        classifier_cfig = self.update_model_config(data, classifier_cfig, layer)
        model = self.obtain_dl_network(classifier_cfig, layer)
        model = self.move_model_to_cuda(model)
        trainer.set_model(model)

        optimizer_cfig = self.obtain_optimizer_cfig()
        optimizer_cfig = self.update_optimizer_config(data, optimizer_cfig, layer)
        optimizer_cfig["parameters"] = model.parameters()
        optimizer = self.obtain_optimizer(optimizer_cfig, layer)
        trainer.set_optim(optimizer)

        # 针对每个分类器, 分别进行进行数据预处理
        est = self._fit(data, trainer, layer)

        if self.debug:
            print("模型的相关配置:")
            print("训练器的配置信息", trainer_cfig)
            print("损失函数的配置信息", loss_fun_cfig)
            print("模型的配置信息", classifier_cfig)
            print("优化器的配置信息", optimizer_cfig)

        return est

    def _fit(self, data, trainer, layer):

        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]

        new_data = self.obtain_new_data(X_train, y_train, X_val, y_val)

        trainer.execute_fit_step(new_data, layer)

        return trainer.obtain_model()

    def move_model_to_cuda(self, model):
        with torch.no_grad():
            if self.cuda:
                return model.cuda()

    def obtain_new_data(self, X_train, y_train, X_val, y_val):
        new_X_train = self.convert_X_to_tensor(X_train)
        new_y_train = self.convert_y_to_tensor(y_train)
        new_X_val = self.convert_X_to_tensor(X_val)
        new_y_val = self.convert_y_to_tensor(y_val)
        new_data = dict(X_train=new_X_train, y_train=new_y_train, X_val=new_X_val, y_val=new_y_val)
        return new_data

    def convert_X_to_tensor(self, Xs):
        new_X = []
        for name in self.m_names:
            X = torch.tensor(Xs[name]).float()
            if self.cuda:
                X = X.cuda()
            new_X.append(X)
        return new_X

    def convert_y_to_tensor(self, y):
        y = torch.tensor(y).long()
        if self.cuda:
            y = y.cuda()
        return y

    def obtain_trainer(self, trainer_cfig, layer):
        trainer_name = trainer_cfig.get("name", None)
        assert trainer_name != None, "训练器的名字不能设置为空"
        trainer = get_dl_trainer(trainer_name, trainer_cfig, layer)
        if trainer == None:
            raise "暂时不支持" + trainer_name + "分类器/特征提取器"
        return trainer

    def obtain_dl_network(self, classifier_cfig, layer):
        assert classifier_cfig != None, "特征提取器的名字不能设置为空"
        est = get_dl_model(classifier_cfig, layer)
        if est == None:
            raise "模型的配置不能为空！"
        return est

    def obtain_loss_fun(self, loss_fun_cfig, layer):
        loss_fun_name = loss_fun_cfig.get("name", None)
        assert loss_fun_name != None, "损失函数的名字不能设置为空"

        if loss_fun_name == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        elif loss_fun_name == "L1Loss":
            return nn.L1Loss()
        elif loss_fun_name == "KLDivLoss":
            return nn.KLDivLoss()
        elif loss_fun_name == "MultiLabelMarginLoss":
            return nn.MultiMarginLoss()

    def obtain_optimizer(self, optimizer_cfig, layer):
        optim_name = optimizer_cfig.get("name", None)
        assert optim_name != None, "优化器的名字不能设置为空"
        parameters = optimizer_cfig.get("parameters")

        if optim_name == "Adam":
            return optim.Adam(parameters)
        elif optim_name == "SGD":
            return optim.SGD(parameters)
        elif optim_name == "RMSprop":
            return optim.RMSprop(parameters)

    def update_trainer_config(self, data, new_cfig, layer):
        return new_cfig

    def update_model_config(self, data, classifier_cfig, layer):
        new_classifier_cfig = copy.deepcopy(classifier_cfig)
        new_classifier_cfig["Layers"] = layer
        return new_classifier_cfig

    def update_loss_fun_config(self, data, new_cfig, layer):
        return new_cfig

    def update_optimizer_config(self, data, new_cfig, layer):
        return new_cfig

    def obtain_trainer_cfig(self):
        return self.trainer_cfig

    def set_trainer_cfig(self, new_trainer_cfig):
        self.trainer_cfig = new_trainer_cfig

    def obtain_classifier_cfig(self):
        return self.classifier_cfig

    def set_classifier_cfig(self, new_classifier_cfig):
        self.classifier_cfig = new_classifier_cfig

    def obtain_loss_fun_cfig(self):
        return self.loss_fun_cfig

    def set_loss_fun_cfig(self, new_loss_fun_cfig):
        self.loss_fun_cfig = new_loss_fun_cfig

    def obtain_optimizer_cfig(self):
        return self.optimizer_cfig

    def set_optimizer_cfig(self, new_optimizer_cfig):
        self.optimizer_cfig = new_optimizer_cfig
