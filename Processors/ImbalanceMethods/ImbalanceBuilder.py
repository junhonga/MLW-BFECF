import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss

def init_sample_methods(sample_method_name, params=None):
    if sample_method_name == "SMOTE":
        return SMOTE(**params)
    if sample_method_name == "RandomOverSampler":
        return RandomOverSampler(**params)
    if sample_method_name == "ADASYN":
        return ADASYN(**params)
    if sample_method_name == "RandomUnderSampler":
        return RandomUnderSampler(**params)
    if sample_method_name == "NearMiss":
        return NearMiss(**params)

class BaseImbalanceBuilder():
    def __init__(self):
        pass

    def fit_resample(self, X_train, y_train):
        pass


class SingleImbalanceBuilder(BaseImbalanceBuilder):

    def __init__(self, configs):
        if isinstance(configs, str):
            name, params = configs, {}
        if isinstance(configs, type):
            name, params = configs
        self.imbalance = init_sample_methods(name, params)

    def fit_resample(self, X, y):
        return self.imbalance.fit_resample(X, y)

class MultiImbalanceBuilder(BaseImbalanceBuilder):
    def __init__(self, configs):
        self.single_imbalance_builder = SingleImbalanceBuilder(configs)

    def fit_resample(self, Xs, y):
        X_resampleds = []
        for X in Xs:
            X_resampled, y_resampled = self.single_imbalance_builder.fit_resample(X,y)
            X_resampleds.append(X_resampled)
        return X_resampleds, y_resampled