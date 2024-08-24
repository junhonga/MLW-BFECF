import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Processors.MetricProcessor.MetricProcessorWrapper import MetricProcessorWrapper


def get_metric_calculator(name, est_type, configs):
    if est_type == "AvgMetricProcessor":
        return AvgMetricProcessor(name, est_type, configs)
    elif est_type == "WeightMetricProcessor":
        return WeightMetricProcessor(name, est_type, configs)
    elif est_type == "MultiAvgMetricProcessor":
        return MultiAvgMetricProcessor(name, est_type, configs)
    elif est_type == "MultiWeightMetricProcessor":
        return MultiWeightMetricProcessor(name, est_type, configs)
    else:
        raise "不支持其他向量拼接方法"

class AvgMetricProcessor(MetricProcessorWrapper):

    def __init__(self, name, est_type, configs):
        super(AvgMetricProcessor, self).__init__(name, est_type, configs)

    def fit_excecute(self, data, layer):
        finfos = data.get("Finfos")
        finfos_layers = self.obtain_finfos_by_layers(finfos, layer)
        need_finfos = self.obtain_need_finfos(finfos_layers)
        need_finfos = self.obtain_need_train_finfos(need_finfos)
        final_probs = self.obtain_probs_by_avg(need_finfos)
        final_preds = np.argmax(final_probs, axis=1)
        y_val = data.get("Original").get("y_val")
        metric = accuracy_score(final_preds, y_val)
        return metric

    def predict_executable(self, finfos, layer):
        return True

    def predict_execute(self, data, layer):
        finfos = data.get("Finfos")
        finfos_layers = self.obtain_finfos_by_layers(finfos, layer)
        need_finfos = self.obtain_need_finfos(finfos_layers)
        need_finfos = self.obtain_need_val_finfos(need_finfos)
        final_probs = self.obtain_probs_by_avg(need_finfos)
        return final_probs

    def obtain_probs_by_avg(self, need_finfos):
        final_probs = []
        for need_finfo in need_finfos:
            final_probs.append(need_finfo.get("Probs"))
        return np.mean(final_probs, axis=0)

class WeightMetricProcessor(MetricProcessorWrapper):

    def __init__(self, name, type, configs):
        super(WeightMetricProcessor, self).__init__(name, type, configs)
        self.finfos_weight_method = configs.get("WeightMethod", "acc")
        self.finfos_weights = None

    def fit_excecute(self, data, layer):
        # 获取预测值需要的数据
        finfos = data.get("Finfos")
        y_val = data.get("Original").get("y_val")

        finfos_layers = self.obtain_finfos_by_layers(finfos, layer)
        need_finfos = self.obtain_need_finfos(finfos_layers)
        need_finfos = self.obtain_need_train_finfos(need_finfos)
        finfo_probs = self.obtain_need_probs(need_finfos)
        # 获得每个预测概率值的权重
        finfos_weight_method = self.obtain_finfos_weight_method()
        finfo_weights = self.calucate_finfos_weight(finfo_probs, y_val, finfos_weight_method)
        self.save_finfos_weights(finfo_weights)
        # 获得最终的预测值, 加权求和
        final_probs = self.obtain_probs_by_weight(need_finfos, finfo_weights)
        #
        final_preds = np.argmax(final_probs, axis=1)
        metric = self.calucate_metric(final_preds, y_val, self.classifier_method)

        return metric

    def obtain_need_probs(self, finfos):
        finfos_probs = []
        for finfo in finfos:
            finfos_probs.append(finfo["Probs_val"])
        return finfos_probs

    def obtain_finfos_weight_method(self):
        return self.finfos_weight_method

    def save_finfos_weights(self, finfo_weights):
        self.finfos_weights = finfo_weights

    def obtain_finfos_weights(self):
        if self.finfos_weights is None:
            raise "预测概率值的权重没有初始化"
        return self.finfos_weights

    def obtain_probs_by_weight(self, need_finfos, finfo_weights):
        final_probs = []
        for need_finfo, finfo_weight in zip(need_finfos, finfo_weights):
            final_probs.append(need_finfo.get("Probs_val") * finfo_weight)
        return np.sum(final_probs, axis=0)

    def calucate_finfos_weight(self, finfo_probs, y_val, weight_method):
        finfo_weights = []
        for finfo in finfo_probs:
            y_pred = np.argmax(finfo, axis=1)
            finfo_weights.append(self.calucate_metric(y_pred, y_val, weight_method))
        finfo_weights = self.normalize_weights(finfo_weights)
        return finfo_weights

    def normalize_weights(self, weights):
        weights_sum = sum(weights)
        weights = [weight / weights_sum for weight in weights]
        return weights

    def predict_execute(self, data, layer):
        finfos = data.get("Finfos")

        finfos_layers = self.obtain_finfos_by_layers(finfos, layer)
        need_finfos = self.obtain_need_finfos(finfos_layers)
        need_finfos = self.obtain_need_val_finfos(need_finfos)

        finfos_weights = self.obtain_finfos_weights()
        final_probs = self.obtain_probs_by_weight(need_finfos, finfos_weights)

        return final_probs

class MultiAvgMetricProcessor(AvgMetricProcessor):

    def __init__(self, name, est_type, configs):
        super(MultiAvgMetricProcessor, self).__init__(name, est_type, configs)

    def fit_excecute(self, data, layer):
        # 获得相关的数据
        finfos = data.get("Finfos")
        y_val = data.get("Original").get("y_val")

        finfos_layers = self.obtain_finfos_by_layers(finfos, layer)
        finfos_by_modalities = self.obtain_finfos_by_modalities(finfos_layers, layer)

        new_unmodal_finfos = []
        for m_name, finfos in finfos_by_modalities.items():
            need_finfos = self.obtain_need_finfos(finfos)
            need_finfos = self.obtain_need_train_finfos(need_finfos)
            need_probs = self.obtain_need_probs(need_finfos)
            unimodal_probs = self.obtain_unimodal_final_probs(need_probs)
            new_unmodal_finfos.append(unimodal_probs)

        new_unmodal_finfos = self.adjuste_new_finfos_by_modalities(new_unmodal_finfos)
        multi_final_probs = self.obtain_multi_final_probs(new_unmodal_finfos)

        final_preds = np.argmax(multi_final_probs, axis=1)
        classifier_method = self.obtain_classifier_method()
        metric = self.calucate_metric(final_preds, y_val, classifier_method)
        return metric

    def obtain_need_probs(self, finfos):
        finfos_probs = []
        for finfo in finfos:
            finfos_probs.append(finfo["Probs_val"])
        return finfos_probs

    def obtain_finfos_by_modalities(self, finfos_layers, layer):
        finfos_by_modalities = dict()

        for finfos_layer in finfos_layers:
            # 获得对应的模态信息
            m_names = finfos_layer.get("ModalityName")
            m_names = "_".join([str(m_name) for m_name in sorted(m_names)])

            finfos = finfos_by_modalities.get(m_names, None)

            if finfos is None:
                finfos_by_modalities[m_names] = list()

            finfos_by_modalities[m_names].append(finfos_layer)

        return finfos_by_modalities

    def obtain_unimodal_final_probs(self, finfo_probs):
        unimodal_probs = np.mean(finfo_probs, axis=0)
        return unimodal_probs

    def adjuste_new_finfos_by_modalities(self, new_finfos_by_modalities):
        return new_finfos_by_modalities

    def obtain_multi_final_probs(self, finfo_probs):
        multi_final_probs = np.mean(finfo_probs, axis=0)
        return multi_final_probs

    def predict_execute(self, data, layer):
        finfos = data.get("Finfos")

        finfos_layers = self.obtain_finfos_by_layers(finfos, layer)
        finfos_by_modalities = self.obtain_finfos_by_modalities(finfos_layers, layer)

        new_unmodal_finfos = []
        for m_name, finfos in finfos_by_modalities.items():
            need_finfos = self.obtain_need_finfos(finfos)
            need_finfos = self.obtain_need_val_finfos(need_finfos)
            need_probs = self.obtain_need_probs(need_finfos)
            unimodal_probs = self.obtain_unimodal_final_probs(need_probs)
            new_unmodal_finfos.append(unimodal_probs)

        final_probs = self.obtain_multi_final_probs(new_unmodal_finfos)

        return final_probs


class MultiWeightMetricProcessor(WeightMetricProcessor):

    def __init__(self, name, est_type, configs):
        super(MultiWeightMetricProcessor, self).__init__(name, est_type, configs)
        self.unimodal_finfos_weight_method = configs.get("UnimodalWeightMethod", "acc")
        self.multi_finfos_weight_method = configs.get("MultiWeightMethod", "acc")
        self.unimodal_finfos_weights = None
        self.multi_finfo_weights = None

    def fit_excecute(self, data, layer):
        # 获得相关的数据
        finfos = data.get("Finfos")
        y_val = data.get("Original").get("y_val")

        finfos_layers = self.obtain_finfos_by_layers(finfos, layer)
        finfos_by_modalities = self.obtain_finfos_by_modalities(finfos_layers, layer)

        new_unmodal_finfos= []
        for m_name, finfos in finfos_by_modalities.items():

            need_finfos = self.obtain_need_finfos(finfos)
            need_probs = self.obtain_need_probs(need_finfos)
            need_finfos = self.obtain_need_train_finfos(need_finfos)
            unimodal_probs, unimodal_finfos_weights = self.obtain_unimodal_final_probs(m_name, need_probs, y_val)
            new_unmodal_finfos.append(unimodal_probs)

        new_unmodal_finfos = self.adjuste_new_finfos_by_modalities(new_unmodal_finfos)
        multi_final_probs, multi_finfo_weights  = self.obtain_multi_final_probs(new_unmodal_finfos, y_val)

        final_preds = np.argmax(multi_final_probs, axis=1)
        classifier_method = self.obtain_classifier_method()
        metric = self.calucate_metric(final_preds, y_val, classifier_method)
        return metric

    def obtain_need_probs(self, finfos):
        finfos_probs = []
        for finfo in finfos:
            finfos_probs.append(finfo["Probs_val"])
        return finfos_probs

    def obtain_finfos_by_modalities(self, finfos_layers, layer):
        finfos_by_modalities = dict()

        for finfos_layer in finfos_layers:
            # 获得对应的模态信息
            m_names = finfos_layer.get("ModalityName")
            m_names = "_".join([str(m_name) for m_name in sorted(m_names)])

            finfos = finfos_by_modalities.get(m_names, None)

            if finfos is None:
                finfos_by_modalities[m_names] = list()

            finfos_by_modalities[m_names].append(finfos_layer)

        return finfos_by_modalities

    def obtain_unimodal_final_probs(self, m_name, need_finfos, y_val):
        unimodal_finfos_weight_method = self.obtain_unimodal_finfos_weight_method()
        unimodal_probs, unimodal_finfos_weights = self.obtain_finfos_dispatcher(need_finfos, unimodal_finfos_weight_method,  y_val)
        self.save_unimodal_finfos_weights(unimodal_finfos_weights)
        return unimodal_probs, unimodal_finfos_weights

    def obtain_unimodal_finfos_weight_method(self):
        return self.unimodal_finfos_weight_method

    def obtain_finfos_dispatcher(self, finfos_by_modalities, finfos_method, y_val=None):
        if finfos_method.lower() in ["avg", "average"]:
            return self.obtain_finfos_by_avg(finfos_by_modalities)
        else:
            finfo_weights = self.obtain_finfo_weights(finfos_by_modalities, finfos_method, y_val)
            final_probs = self.obtain_probs_by_weight(finfos_by_modalities, finfo_weights)
        return final_probs, finfo_weights

    def obtain_finfos_by_avg(self, finfos_by_modalities):
        final_finfos = []
        for need_finfo in finfos_by_modalities:
            final_finfos.append(need_finfo.get("Probs"))
        return np.mean(final_finfos, axis=0)

    def obtain_finfo_weights(self, finfos_by_modalities, finfos_method, y_val):
        finfo_weights = []
        for finfo in finfos_by_modalities:
            y_pred = np.argmax(finfo, axis=1)
            finfo_weights.append(self.calucate_metric(y_pred, y_val, finfos_method))
        finfo_weights = self.normalize_weights(finfo_weights)
        return finfo_weights

    def save_unimodal_finfos_weights(self, unimodal_finfos_weights):
        self.unimodal_finfos_weights = unimodal_finfos_weights

    def adjuste_new_finfos_by_modalities(self, new_finfos_by_modalities):
        return new_finfos_by_modalities

    def obtain_multi_final_probs(self, need_finfos, y_val):
        multi_finfos_weight_method = self.obtain_multi_finfos_weight_method()
        multi_final_probs, multi_finfo_weights = self.obtain_finfos_dispatcher(need_finfos, multi_finfos_weight_method, y_val)
        self.save_multi_finfos_weights(multi_finfo_weights)
        return multi_final_probs, multi_finfo_weights

    def save_multi_finfos_weights(self, multi_finfo_weights):
        self.multi_finfo_weights = multi_finfo_weights

    def obtain_multi_finfos_weight_method(self):
        return self.multi_finfos_weight_method

    def obtain_multi_weights(self):
        if self.multi_finfo_weights is None:
            raise "多模态的权重没有设置"
        return self.multi_finfo_weights

    def obtain_probs_by_weight(self, finfo_probs, weights):
        final_finfo_probs = []
        for finfo, weight in zip(finfo_probs, weights):
            final_finfo_probs.append(finfo * weight)
        return np.sum(final_finfo_probs, axis=0)

    def predict_execute(self, data, layer):
        finfos = data.get("Finfos")

        finfos_layers = self.obtain_finfos_by_layers(finfos, layer)
        finfos_by_modalities = self.obtain_finfos_by_modalities(finfos_layers, layer)

        new_unmodal_finfos = []
        for m_name, finfos in finfos_by_modalities.items():
            need_finfos = self.obtain_need_finfos(finfos)
            need_probs = self.obtain_need_probs(need_finfos)
            need_finfos = self.obtain_need_val_finfos(need_finfos)
            unimodal_finfo_weights = self.obtain_unimodal_weights()
            final_probs = self.obtain_probs_by_weight(need_probs, unimodal_finfo_weights)
            new_unmodal_finfos.append(final_probs)

        multi_finfo_weights = self.obtain_multi_weights()
        final_probs = self.obtain_probs_by_weight(new_unmodal_finfos, multi_finfo_weights)

        return final_probs

    def obtain_unimodal_weights(self):
        if self.unimodal_finfos_weights is None:
            raise "单模态的权重没有设置"
        return self.unimodal_finfos_weights

    def obtain_multi_weights(self):
        if self.multi_finfo_weights is None:
            raise "多模态的权重没有设置"
        return self.multi_finfo_weights
