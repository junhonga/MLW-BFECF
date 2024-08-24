import copy

from ClassifierBuider.ClassifierBuilder import get_classifier_buider, BuilderBuiderWrapper
from Processors.ProcessorDispatcher.ProcessorsDispatcherWrapper import Dispatcher


class ClassifierDispatcher(Dispatcher):

    def __init__(self):
        super(ClassifierDispatcher, self).__init__(BuilderBuiderWrapper)

    def obtain_instance(self, configs):
        classifier_builders = []
        for config in configs:
            builder_name = config.get("BuilderName", None)
            m_names = config.get("ModalityNames", None)
            builder_type = config.get("BuilderType", None)
            new_config = copy.deepcopy(config)
            for m_name in m_names:
                new_config["ModalityNames"] = m_name
                # 获得新的分类器构建器列表, 不过该构建器是一个列表,
                classifier_builder_list = self.obtain_builder_dispatcher(builder_name, builder_type, config, m_name, new_config)
                # 添加新的分类器构建器
                if classifier_builder_list != None and len(classifier_builder_list) > 0:
                    classifier_builders.extend(classifier_builder_list)

        return classifier_builders

    def obtain_builder_dispatcher(self, builder_name, builder_type, config, m_name, new_config):
        if builder_type == "ML":
            classifier_builder = self.obtain_ML_builder(builder_name, builder_type, m_name, new_config, config)
        elif builder_type == "DL":
            classifier_builder = self.obtain_DL_builder(builder_name, builder_type, m_name, new_config, config)
        return classifier_builder

    def obtain_ML_builder(self, builder_name, builder_type, m_name, new_config, config):
        classifier_builders = []
        classifier_configs = config.get("ClassifierConfig", None)
        #
        if isinstance(classifier_configs, list):
            for classifier_config in classifier_configs:
                # 获得对应的分类器
                new_config["ClassifierConfig"] = classifier_config
                classifier_builder = self.execute_dispatcher_method(builder_name, builder_type, m_name, new_config)
                self.check_builder(classifier_builder)
                classifier_builders.append(classifier_builder)
        return classifier_builders

    def obtain_DL_builder(self, builder_name, builder_type, m_name, new_config, config):
        classifier_builders = []
        model_configs = config.get("Model", None)
        #
        if isinstance(model_configs, list):
            for classifier_config in model_configs:
                # 获得对应的分类器
                new_config["Model"] = classifier_config
                classifier_builder = self.execute_dispatcher_method(builder_name, builder_type, m_name, new_config)
                self.check_builder(classifier_builder)
                classifier_builders.append(classifier_builder)
        return classifier_builders

    def check_builder(self, classifier_builder):
        if not isinstance(classifier_builder, self.Template):
            raise classifier_builder + "没有继承 " + self.Template + " 类"

    def obtain_builder_name(self, name, m_names):
        return name + "&" + "_".join([str(m_name) for m_name in m_names])

    def execute_dispatcher_method(self, builder_name, builder_type, modality_names, config):
        new_cfig = copy.deepcopy(config)
        return get_classifier_buider(builder_name, builder_type, modality_names, new_cfig)