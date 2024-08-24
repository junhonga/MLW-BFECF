class ClassifierTemplate():

    def __init__(self, est_type, configs, layer):
        self.est_type = est_type
        self.layer = layer
        self.name = configs.get("Name", None)
        self.m_names = configs.get("ModalityNames", None)
        self.modality_num = len(self.m_names)
        self.builder_type = configs.get("BuilderType", None)
        self.classifier_type = configs.get("ClassifierType", None)
        self.processors = configs.get("Processors", [])
        self.classifier_instance = None
        self.is_probs_executable = True
        self.is_predict_executable = True
        self.is_fit_executable = True
        self.is_features_executable = True

    def obtain_is_probs_executable(self):
        return self.is_probs_executable

    def set_is_probs_executable(self, is_probs_executable):
        self.is_probs_executable = is_probs_executable

    def obtain_is_predict_executable(self):
        return self.is_predict_executable

    def set_is_predict_executable(self, is_predict_executable):
        self.is_predict_executable = is_predict_executable

    def obtain_is_fit_executable(self):
        return self.is_fit_executable

    def set_is_fit_executable(self, is_fit_executable):
        self.is_fit_executable = is_fit_executable

    def obtain_is_features_executable(self):
        return self.is_features_executable

    def set_is_features_executable(self, is_features_executable):
        self.is_features_executable = is_features_executable

    def execute_pre_fit_processors(self, data, layer):
        # 获得分类器实例
        cls_instance = self.obtain_classifier_instance()
        # 训练的前置处理器
        for pre_processor in self.processors:
            if pre_processor.pre_executable(layer):
                data, cls_instance = pre_processor.pre_fit_execute(data, cls_instance, layer)
        # 将分类器实例重新设置回去
        self.set_classifier_instance(cls_instance)
        return data

    def execute_post_fit_processors(self, data, layer):
        # 获得分类器实例
        cls_instance = self.obtain_classifier_instance()
        # 训练的后置处理器
        for pre_processor in self.processors:
            if pre_processor.post_executable(layer):
                data, cls_instance = pre_processor.post_fit_execute(data, cls_instance, layer)
        # 将分类器实例重新设置回去
        self.set_classifier_instance(cls_instance)

    def execute_fit_step(self, data, layer):
        data = self.execute_pre_fit_processors(data, layer)
        self._execute_fit_step(data, layer)
        self.execute_post_fit_processors(data, layer)

    def _execute_fit_step(self, data, layer):
        # 获得训练集
        Xs_train, y_train = data["X_train"], data["y_train"]
        Xs_val, y_val = data.get("X_val", None), data.get("y_val", None)
        # 训练训练器
        self.fit(Xs_train, y_train, Xs_val, y_val)

    def fit(self, X_train, y_train, X_val, y_val):
        pass

    def execute_pre_predict_probs_processors(self, data, layer):
        # 获得分类器实例
        cls_instance = self.obtain_classifier_instance()
        # 训练的前置处理器
        for pre_processor in self.processors:
            if pre_processor.pre_executable(layer):
                data, cls_instance = pre_processor.pre_predict_probs_execute(data, cls_instance, layer)

        return data

    def execute_post_predict_probs_processors(self, data, layer):
        # 获得分类器实例
        cls_instance = self.obtain_classifier_instance()
        # 训练的后置处理器
        for pre_processor in self.processors:
            if pre_processor.post_executable(layer):
                data, cls_instance = pre_processor.post_predict_probs_execute(data, cls_instance, layer)

    def execute_predict_probs_step(self, data, layer):
        data = self.execute_pre_predict_probs_processors(data, layer)
        self._execute_predict_probs_step(data, layer)
        self.execute_post_predict_probs_processors(data, layer)

    def _execute_predict_probs_step(self, data, layer):
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data.get("X_val", None), data.get("y_val", None)
        self.fit(X_train, y_train, X_val, y_val)

    def predict_probs(self, X):
        pass

    def predict(self, X):
        pass

    def execute_pre_obtain_features_processors(self, data, layer):
        # 获得分类器实例
        cls_instance = self.obtain_classifier_instance()
        # 训练的前置处理器
        for pre_processor in self.processors:
            if pre_processor.pre_executable(layer):
                data, cls_instance = pre_processor.pre_obtain_features_execute(data, cls_instance, layer)

        return data

    def execute_post_obtain_features_processors(self, data, layer):
        # 获得分类器实例
        cls_instance = self.obtain_classifier_instance()
        # 训练的后置处理器
        for pre_processor in self.processors:
            if pre_processor.post_executable(layer):
                data, cls_instance = pre_processor.post_obtain_features_execute(data, cls_instance, layer)

    def execute_obtain_features_step(self, data, layer):
        data = self.execute_pre_obtain_features_processors(data, layer)
        self._execute_obtain_features_step(data, layer)
        self.execute_post_obtain_features_processors(data, layer)

    def _execute_obtain_features_step(self, data, layer):
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data.get("X_val", None), data.get("y_val", None)
        self.fit(X_train, y_train, X_val, y_val)

    def obtain_features(self, X):
        return None

    def can_obtain_probs(self, layer):
        return self.is_probs_executable

    def can_obtain_predict(self, layer):
        return self.is_predict_executable

    def can_obtain_features(self, layer):
        return self.is_features_executable

    def fit_executable(self, infos, layer):
        return self.is_fit_executable

    def obtain_name(self):
        return self.name

    def obtain_layers(self):
        return self.layer

    def obtain_est_type(self):
        return self.est_type

    def obtain_builder_type(self):
        return self.builder_type

    def obtain_classifier_instance(self):
        return self.classifier_instance

    def set_classifier_instance(self, classifier_instance):
        self.classifier_instance = classifier_instance

    def obtain_modality_name(self):
        return self.m_names

    def set_modality_name(self, m_names):
        self.m_names = m_names
        self.m_names_num = len(m_names)

    def obtain_classifier_type(self):
        return self.classifier_type
