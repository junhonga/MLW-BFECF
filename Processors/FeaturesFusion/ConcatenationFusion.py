import numpy as np

from Common.Template import RootTemplate


def get_feature_concatenation_method(name, est_type, configs):
    if est_type == "FeatureConcatenation":
        return FeatureConcatenation(name, type, configs)
    else:
        raise "不支持其他向量拼接方法"

class FeaturesFusionTemplate(RootTemplate):
    pass

class FeatureConcatenation(FeaturesFusionTemplate):

    def __init__(self, name, est_type, configs):
        self.name = name
        self.est_type = est_type
        self.layers = configs.get("Layers", 1)
        self.builder_type = configs.get("BuilderType", [])
        self.est_type = configs.get("EstType", [])
        self.components = dict()

    def executable(self, layer):
        return layer >= 2

    def fit_excecute(self, original_train, original_val, finfos, modality_name, layer):

        finfos_layers = self.obtain_finfos_by_layers(finfos, layer)
        finfos_layers = self.delete_missing_fit_finfos(finfos_layers)
        need_finfos = self.obtain_need_finfos(finfos_layers, modality_name)
        features_train, features_val = self.obtain_final_fit_features(need_finfos)
        if features_train is not None:
            fusions_train = np.concatenate([original_train, features_train] , axis=1)
            fusions_val = np.concatenate([original_val, features_val], axis=1)
        else:
            fusions_train, fusions_val = original_train, original_val

        return fusions_train, fusions_val

    def predict_excecute(self, original_X, finfos, modality_name, layer):

        finfos_layers = self.obtain_finfos_by_layers(finfos, layer)
        finfos_layers = self.delete_missing_predict_finfos(finfos_layers)
        need_finfos = self.obtain_need_finfos(finfos_layers, modality_name)
        features_X = self.obtain_final_predict_features(need_finfos)
        if features_X is not None:
            fusions_X = np.concatenate([original_X, features_X] , axis=1)
        else:
            fusions_X = original_X

        return fusions_X

    def obtain_finfos_by_layers(self, finfos, layer):
        need_layer = layer - 1
        return finfos.get(need_layer)

    def delete_missing_fit_finfos(self, finfos_layers):
        need_finfos = []
        for finfo in finfos_layers:
            if finfo.get("Feature_train") is None:
                continue
        return need_finfos

    def obtain_final_fit_features(self, need_finfos):
        if len(need_finfos) > 0 :
            final_features_train, final_features_val = [], []
            for need_finfo in need_finfos:
                final_features_train.append(need_finfo.get("Feature_train"))
                final_features_val.append(need_finfo.get("Feature_val"))
            final_features_train = np.concatenate(final_features_train, axis=1)
            final_features_val = np.concatenate(final_features_val, axis=1)
            return final_features_train, final_features_val
        else:
            final_features_train, final_features_val = None, None
            return final_features_train, final_features_val

    def obtain_final_predict_features(self, need_finfos):
        if len(need_finfos) > 0 :
            final_features_X = []
            for need_finfo in need_finfos:
                final_features_X.append(need_finfo.get("Feature_X"))
            final_features_X = np.concatenate(final_features_X, axis=1)
            return final_features_X
        else:
            final_features_X = None
            return final_features_X

    def delete_missing_predict_finfos(self, finfos_layers):
        need_finfos = []
        for finfo in finfos_layers:
            if finfo.get("Feature_X") is None:
                continue
        return need_finfos

    def obtain_need_finfos(self, finfos_layers, modality_name):
        need_finfos = []
        for finfo in finfos_layers:
            if finfo.get("BuilderType") in self.builder_type and modality_name in finfo.get("ModalityName"):
                need_finfos.append(finfo)
                continue
            if finfo.get("EstType") in self.est_type and modality_name in finfo.get("ModalityName"):
                need_finfos.append(finfo)
                continue
        return need_finfos
