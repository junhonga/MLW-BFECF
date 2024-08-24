import copy

import torch
import numpy as np

from Classification.CommonTemplate.ClassifierTemplate import ClassifierTemplate


class ModelWrapper(ClassifierTemplate):

    def __init__(self, model_type, est_class, configs, layer):
        super(ModelWrapper, self).__init__(model_type, configs, layer)
        self.est_class = est_class
        self.est_args = configs.get("Parameter", None)
        self._init_estimator(est_class, self.est_args)

    def _init_estimator(self, est_class, est_args):
        self.est = est_class(**est_args)

    def predict(self, X):
        return np.argmax(self.predict_probs(X), axis=1)

    def predict_probs(self, X):
        with torch.no_grad():
            X = self.obtain_Xs_by_modality(X)
            X = self.convert_X_to_tensor(X)
            _, outputs = self.est(*X)
            outputs = self.convert_data_to_numpy(outputs)
            return outputs

    def obtain_Xs_by_modality(self, Xs):
        new_Xs = []
        for m_name in self.m_names:
            X = Xs[m_name]
            new_Xs.append(X)
        return new_Xs

    def convert_X_to_tensor(self, Xs):
        new_Xs = []
        for X in Xs:
            X = torch.tensor(X).float()
            if self.cuda:
                X = X.cuda()
            new_Xs.append(X)

        return new_Xs

    def convert_data_to_numpy(self, X):
        return X.cpu().detach().numpy() if X.device != "cpu" else X.detach().numpy()

    def obtain_features(self, X):
        with torch.no_grad():
            X = self.obtain_Xs_by_modality(X)
            new_X = self.convert_X_to_tensor(X)
            features, _ = self.est(*new_X)
            features = self.convert_data_to_numpy(features)
            return features

    def __call__(self, *args, **kwargs):
        return self.est(*args, **kwargs)

    def parameters(self):
        return self.est.parameters()

    def cuda(self):
        self.est = self.est.cuda()
        return self

    def obtain_instance(self):
        return self.est
