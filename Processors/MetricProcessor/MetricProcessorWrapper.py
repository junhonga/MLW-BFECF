from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Common.Template import RootTemplate

class MetricProcessorTemplate(RootTemplate):
    pass

class MetricProcessorWrapper(MetricProcessorTemplate):

    def __init__(self, name, est_type, configs):
        self.name = name
        self.type = est_type
        self.can_fit_executable = True
        self.builder_type = configs.get("BuilderType", [])
        self.est_type = configs.get("EstType", [])
        self.data_type = configs.get("DateType", [])
        self.modality_name = configs.get("ModalityName")
        self.classifier_method = configs.get("MetricMethod", "acc")
        self.use_layer_type = configs.get("UseLayerType", 'NearestLayer')
        if self.use_layer_type == "NearestLayer":
            self.max_use_layer_num = configs.get("UseLayerNum", 1)
        assert self.classifier_method != None, "分类器的方法设置不能为空"

    def executable(self, layer):
        return self.can_fit_executable

    def obtain_need_finfos(self, finfos_layers):
        need_finfos = []
        for finfo in finfos_layers:
            if finfo.get("BuilderType") in self.builder_type:
                need_finfos.append(finfo)
                continue
            if finfo.get("EstType") in self.est_type:
                need_finfos.append(finfo)
                continue
            if finfo.get("DataType") in self.data_type:
                need_finfos.append(finfo)
                continue
        return need_finfos

    def obtain_need_train_finfos(self, need_finfos):
        new_need_finfos = []
        for finfo in need_finfos:
            if finfo.get("Probs_val") is None:
                continue
            new_need_finfos.append(finfo)
        return new_need_finfos

    def obtain_need_val_finfos(self, need_finfos):
        new_need_finfos = []
        for finfo in need_finfos:
            if finfo.get("Probs_val") is None:
                continue
            new_need_finfos.append(finfo)
        return new_need_finfos

    def obtain_finfos_by_layers(self, finfos, layer):
        # 这个方法进入的是字典， 但是返回的时候应该为列表, 因为每层中都有相同名字的
        if self.use_layer_type == "NearestLayer":
            finfos_list = []
            max_use_layer_num = self.obtain_max_use_layer_num()
            for i in range(layer, layer - max_use_layer_num, -1):
                # 保留层数的索引号不可能低于 0
                if i <= 0:
                    break
                finfos_list.extend(list(finfos.get(i)))
            return finfos_list

    def calucate_metric(self, x1, x2, method_name):
        if isinstance(self.classifier_method, str):
            if method_name.lower() in ["accuracy", "acc"]:
                return accuracy_score(x1, x2)
            if method_name.lower() in ["precision", "pre"]:
                return precision_score(x1, x2)
            if method_name.lower() in ["recall"]:
                return recall_score(x1, x2)
            if method_name.lower() in ["f1_score", "f1", "f1-score"]:
                return f1_score(x1, x2)
        elif callable(self.classifier_method):
            return self.classifier_method(x1, x2)

    def obtain_classifier_method(self):
        return self.classifier_method

    def set_classifier_method(self, classifier_method):
        self.classifier_method = classifier_method

    def obtain_can_executable(self):
        return self.can_fit_executable

    def set_can_executable(self, can_fit_executable):
        self.can_fit_executable = can_fit_executable

    def set_max_use_layer_num(self, max_use_layer_num):
        self.max_use_layer_num = max_use_layer_num

    def obtain_max_use_layer_num(self):
        return self.max_use_layer_num
