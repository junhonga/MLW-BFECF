import logging

import numpy as np
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

from Classification.CommonTemplate.ClassifierTemplate import ClassifierTemplate
from Classification.MLClassifier.BaseClassify import get_ml_base_classifier


def get_ens_classifier(est_type, configs, layer, default=False):
    if est_type == "EnsembleClassify":
        return EnsembleClassify(est_type, configs, layer)
    elif est_type == "WeightEnsembleClassify":
        return WeightEnsembleClassify(est_type, configs, layer)
    elif est_type == "AdaptiveEnsembleClassifyByNum":
        return AdaptiveEnsembleClassifyByNum(est_type, configs, layer)
    elif est_type == "AdaptiveEnsembleClassifyByMid":
        return AdaptiveEnsembleClassifyByMid(est_type, configs, layer)
    elif est_type == "AdaptiveEnsembleClassifyByAvg":
        return AdaptiveEnsembleClassifyByAvg(est_type, configs, layer)
    else:
        if default:
            return None
        else:
            raise "暂时不支持" + est_type + "分类器"

class EnsembleClassify(ClassifierTemplate):

    def __init__(self, est_type, configs, layer):

        super(EnsembleClassify, self).__init__(est_type, configs, layer)

        self.BaseClassifierConfig = configs.get("BaseClassifier", None)
        assert self.BaseClassifierConfig != None, "基分类器必须配置"

        self.init_base_classifiers(layer)
        assert self.BaseClassifierNum != 0, "基分类器的数量不能为空！"

        self.is_encapsulated = configs.get("IsEncapsulated", True)

        if self.is_encapsulated:
            print("使用的集成分类器的名称:", self.name)
            print("初始化的分类器数量:", self.BaseClassifierNum)
            print("初始化的基分类器名称", self.obtain_est_name())

    def init_base_classifiers(self, layer):
        base_classifier_names, base_classifier_intances = self.init_base_classifiers_instance(layer)

        self.BaseClassifierNames = base_classifier_names
        self.BaseClassifierIntances = base_classifier_intances

        self.BaseClassifierNum = len(self.BaseClassifierIntances)

    def init_base_classifiers_instance(self, layer):
        base_classifier_names = []
        base_classifier_intances = []
        for config in self.BaseClassifierConfig:
            if self.check_classifier_init(config, layer):
                #
                est_name = config.get("ClassifierName", None)
                est_type = config.get("ClassifierType", None)
                modality_names = config.get("ModalityNames", None)
                if modality_names is None:
                    config["ModalityNames"] = self.obtain_modality_name()

                base_classifier_names.append(est_name)
                base_classifier_intances.append(get_ml_base_classifier(est_name, est_type, config, layer))

        return base_classifier_names, base_classifier_intances

    def fit(self, X_train, y_train, X_test=None, y_test=None):

        for est in self.BaseClassifierIntances:
            est.fit(X_train, y_train)

    def obtain_new_X(self, X):
        m_names = self.obtain_modality_name()
        m_names_num = len(m_names)
        if m_names_num == 1:
            return X[m_names[0]]
        elif m_names_num > 1:
            new_X = [X[m_name] for m_name in m_names]
            new_X = np.concatenate(new_X, axis=1)
            return new_X

    def predict_all_probs(self, X):
        probs = list()
        for est in self.BaseClassifierIntances:
            proba = est.predict_proba(X)
            probs.append(proba)
        return probs

    def predict_probs(self, X):
        X = self.obtain_new_X(X)
        probs = self.predict_all_probs(X)
        return np.mean(np.stack(probs), axis=0)

    def predict(self, X):
        return np.argmax(self.predict_probs(X), axis=1)

    def obtain_features(self, X):
        features = self.predict_all_probs(X)
        return np.concatenate(features, axis=1)

    def obtain_est_name(self):
        return self.BaseClassifierNames

    def obtain_classifier_instance(self):
        return self.BaseClassifierIntances

    def check_classifier_init(self, config, layer):
        # 这个方法是用于判断是否当前层需要初始化哪些基分类器
        need_layers = config.get("Layer", None)
        if need_layers == None:
            return True
        if layer in need_layers:
            return True
        return False

class WeightEnsembleClassify(EnsembleClassify):

    def __init__(self, est_type, configs, layer):
        super(WeightEnsembleClassify, self).__init__(est_type, configs, layer)

        self.weight_method = configs.get("WeightMetric", "acc")

        self.ClassifierWeights = {}


    def _calculate_weight_metric(self, X_test, y_test, est):
        if callable(self.weight_method):
            return self.weight_method(X_test, y_test, est)
        elif isinstance(self.weight_method, str):
            y_pre = est.predict(X_test)
            return self.obtain_built_in_weight_method(y_pre, y_test)
        else:
            raise "出错"

    def normalize_weights(self, weights):
        weights_sum = sum(weights.values())
        for name, _ in weights.items():
            weights[name] = weights[name] / weights_sum
        return weights

    def calculate_weight_metrics(self, X_test, y_test):
        weight_metrics = {}
        for est in self.BaseClassifierIntances:
            est_name = est.obtain_name()
            weight_metrics[est_name] = self._calculate_weight_metric(X_test, y_test, est)
        return self.normalize_weights(weight_metrics)

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        super(WeightEnsembleClassify, self).fit(X_train, y_train, X_test=None, y_test=None)
        self.ClassifierWeights = self.calculate_weight_metrics(X_test, y_test)

    def predict_probs(self, X):
        features = self.predict_all_probs(X)
        return self.obtain_probs_by_weight(features)

    def obtain_features(self, X):
        features = self.predict_all_probs(X)
        return self.obtain_probs_by_weight(features)

    def obtain_weights(self):
        return self.ClassifierWeights

    def obtain_probs_by_weight(self, features):
        return np.sum([weight * features[name] for name, weight in self.ClassifierWeights.items()], axis=0)

    def obtain_built_in_weight_method(self, x1, x2):
        if self.weight_method.lower() in ["accuracy", "acc"]:
            return accuracy_score(x1, x2)
        if self.weight_method.lower() in ["precision", "pre"]:
            return precision_score(x1, x2)
        if self.weight_method.lower() in ["recall"]:
            return recall_score(x1, x2)
        if self.weight_method.lower() in ["f1_score", "f1", "f1-score"]:
            return f1_score(x1, x2)

    def set_weight_method(self, weight_method):
        if callable(weight_method):
            self.weight_method = weight_method
        else:
            logging.error("设置的分类器必须要是可调用的")

class AdaptiveEnsembleClassify(EnsembleClassify):

    def __init__(self, est_type, configs, layer):
        super(AdaptiveEnsembleClassify, self).__init__(est_type, configs, layer)

        self.RetainedClassifierName = None
        self.RetainedClassifier = None
        self.ClassifierMetrics = {}

        self.metric_method = configs.get("CaluateMetric", "acc")

    def calculate_adaptive_metrics(self, X_test, y_test):
        classifier_metrics = {}
        for est in self.BaseClassifierIntances:
            est_name = est.obtain_name()
            classifier_metrics[est_name] = self._calculate_adaptive_metric(X_test, y_test, est)
        return classifier_metrics

    def print_classifier_metrics(self):
        print("当前层不同分类器的指标:", end=" ")
        for name, metric in self.ClassifierMetrics.items():
            print(name, ":", format(metric, ".4f"), end=", ")
        print()

    def obtain_retained_classifier_instance_by_name(self, retained_classifier_name):
        retained_ests = []
        for base_classifier_instance in self.BaseClassifierIntances:
            est_name = base_classifier_instance.obtain_name()
            if est_name in retained_classifier_name:
                retained_ests.append(base_classifier_instance)
        return retained_ests


    def fit(self, X_train, y_train, X_test=None, y_test=None):
        super(AdaptiveEnsembleClassify, self).fit(X_train, y_train, X_test=None, y_test=None)
        classifier_metrics = self.calculate_adaptive_metrics(X_test, y_test)
        self.set_classifier_metrics(classifier_metrics)
        # 获得保留的分类器名字
        retained_classifier_names = self.complete_adaptive_method()
        self.set_retained_classifier_name(retained_classifier_names)
        # 通过分类器名字获得具体的分类器
        retained_classifiers = self.obtain_retained_classifier_instance_by_name(retained_classifier_names)
        self.set_retained_classifier(retained_classifiers)

        if self.is_encapsulated:
            self.print_classifier_metrics()
            print("筛选出的基分类器:", retained_classifier_names)

    def predict_retained_probs(self, X):
        features = []
        X = self.obtain_new_X(X)
        for est in self.RetainedClassifier:
            features.append(est.predict_proba(X))
        return features

    def predict_probs(self, X):
        probs = self.predict_retained_probs(X)
        return np.mean(np.stack(probs), axis=0)

    def obtain_features(self, X):
        features = self.predict_retained_probs(X)
        return np.concatenate(features, axis=1)

    def obtain_retained_ens_name(self):
        est_names_temp = []
        for is_retained, est_name in zip(self.retained, self.ensemble_names):
            if is_retained:
                est_names_temp.append(est_name)
        return est_names_temp

    def obtain_retained_ens_instances(self):
        est_instances_temp = []
        for is_retained, est in zip(self.retained, self.ensembles):
            if is_retained:
                est_instances_temp.append(est)
        return est_instances_temp

    def _calculate_adaptive_metric(self, X_test, y_test, ens=None):
        if callable(self.metric_method):
            return self.metric_method(X_test, y_test, ens)
        elif isinstance(self.metric_method, str):
            y_pre = ens.predict(X_test)
            return self.obtain_built_in_metric(y_pre, y_test)
        else:
            raise "出错"


    def set_metric_method(self, metric_method):
        if callable(metric_method):
            self.metric_method = metric_method
        else:
            logging.error("设置的分类器必须要是可调用的")

    def complete_adaptive_method(self):
        pass

    def obtain_built_in_metric(self, x1, x2):
        if self.metric_method.lower() in ["accuracy", "acc"]:
            return accuracy_score(x1, x2)
        if self.metric_method.lower() in ["precision", "pre"]:
            return precision_score(x1, x2)
        if self.metric_method.lower() in ["recall"]:
            return recall_score(x1, x2)
        if self.metric_method.lower() in ["f1_score", "f1", "f1-score"]:
            return f1_score(x1, x2)

    def set_classifier_metrics(self, classifier_metrics):
        self.ClassifierMetrics = classifier_metrics

    def obtain_classifier_metrics(self):
        return self.ClassifierMetrics

    def set_retained_classifier_name(self, retained_classifier_name):
        self.RetainedClassifierName = retained_classifier_name

    def obtain_retained_classifier_name(self):
        return self.RetainedClassifierName

    def set_retained_classifier(self, retained_classifier):
        self.RetainedClassifier = retained_classifier
        self.retained_num = len(retained_classifier)

    def obtain_retained_classifier(self):
        return self.RetainedClassifier

class AdaptiveEnsembleClassifyByNum(AdaptiveEnsembleClassify):

    def __init__(self, est_type, configs, layer):

        super(AdaptiveEnsembleClassifyByNum, self).__init__(est_type, configs, layer)

        self.retained_num = configs.get("RetainedNum", 3)
        print("筛选后保留的基分类器的数量:", self.retained_num)

        assert len(self.BaseClassifierIntances) >= self.retained_num, "基分类器的数量必须要大于保留的数量"

    def complete_adaptive_method(self):
        classifier_metrics = self.obtain_classifier_metrics()
        return sorted(classifier_metrics, key=classifier_metrics.get, reverse=True)[:self.retained_num]

    def set_retained_num(self, retained_num):
        self.retained_num = retained_num

    def obtain_retained_num(self):
        return self.retained_num

class AdaptiveEnsembleClassifyByMid(AdaptiveEnsembleClassify):

    def __init__(self, est_type, configs, layer):
        super(AdaptiveEnsembleClassifyByMid, self).__init__(est_type, configs, layer)

    def complete_adaptive_method(self):
        classifier_metrics = self.obtain_classifier_metrics()
        mid = np.median(list(classifier_metrics.values()))
        # 获得保留的分类器名字
        retained_classifier_name = [name for name, metric in self.ClassifierMetrics.items() if metric > mid]
        return retained_classifier_name

class AdaptiveEnsembleClassifyByAvg(AdaptiveEnsembleClassify):

    def __init__(self, est_type, configs, layer):
        super(AdaptiveEnsembleClassifyByAvg, self).__init__(est_type, configs, layer)

    def complete_adaptive_method(self):
        classifier_metrics = self.obtain_classifier_metrics()
        mn = np.mean(list(classifier_metrics.values()))
        retained_classifier_name = [name for name, metric in self.ClassifierMetrics.items() if metric > mn]
        return retained_classifier_name

class AdaptiveWeightEnsembleClassify(AdaptiveEnsembleClassify, WeightEnsembleClassify):

    def __init__(self, est_type, configs, layer):
        EnsembleClassify.__init__(self, est_type, configs, layer)

        self.RetainedClassifier = None
        self.ClassifierWeights = {}

        self.ClassifierMetrics = {}
        self.weight_method = configs.get("WeightMetric", "acc")
        self.metric_method = configs.get("CaluateMetric", "acc")

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        EnsembleClassify.fit(X_train, y_train, X_test=None, y_test=None)
        self.ClassifierMetrics = self.calculate_adaptive_metrics(X_test, y_test)
        self.RetainedClassifier = self.complete_adaptive_method(self.ClassifierMetrics)
        self.ClassifierWeights = self.calculate_weight_metrics(X_test, y_test, self.RetainedClassifier)
        if self.is_encapsulated:
            self.print_classifier_metrics()
            print("筛选出的基分类器:", self.RetainedClassifier)

    def predict_proba(self, X):
        probs = self.predict_retained_probs(X)
        return self.obtain_probs_by_weight(probs)

    def obtain_features(self, X):
        features = self.predict_retained_probs(X)
        return self.obtain_probs_by_weight(features)

    def calculate_weight_metrics(self, X_test, y_test, retained_classifier):
        weight_metrics = {}
        for name in retained_classifier:
            est = self.BaseClassifierIntances[name]
            weight_metrics[name] = self._calculate_weight_metric(X_test, y_test, est)
        return self.normalize_weights(weight_metrics)
