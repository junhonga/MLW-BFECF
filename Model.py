import copy
import time
import warnings

import numpy as np

from ClassifierBuider.ClassifierBuilderDispatcher import ClassifierDispatcher
from Processors.ProcessorDispatcher.ProcessorsDispatcher import MetricProcessorDispatcher, FeaturesProcessorDispatcher, \
    MultiFeatureSelectorDispatcher, MultiPreProcessorDispatcher, MultiFusionFeatDispatcher, \
    MultiPostProcessorDispatcher, CategoryImbalanceDispatcher, MultiFeatureFusionDispatcher

warnings.filterwarnings("ignore")

class MultiModalityModel():

    def __init__(self, config):
        assert config != None, "多模态级联模型的配置信息不能为空"
        self.config = config

        self.max_num_iterations = config.get("MaxNumIterations", 20)
        self.termination_layer = config.get("TerminationLayer", 3)
        self.class_num = config.get("ClassNum", None)
        self.debug = config.get("DeBug", True)

        self.classifier_instances = dict()
        self.multi_fselect_ids = {}

        self._init_components(config)

        self.all_feature_fusions_processors = dict()
        self.all_fusion_feature_processors = dict()
        self.all_feature_processors = dict()
        self.all_metrics_processors = dict()

    def _init_components(self, config):
        self._init_pre_processors(config)
        self._init_pre_cascade_processors(config)
        self._init_feature_selectors(config)
        self._init_data_and_feature_fusion(config)
        self._init_fusion_feature_processors(config)
        self._init_category_imbalance_processors(config)
        self._init_cascade_classifier_builder(config)
        self._init_cascade_features_processors(config)
        self._init_cascade_metrics_processors(config)
        self._init_post_cascade_processors(config)
        self._init_post_processor(config)

    def _init_pre_processors(self, configs):
        multi_pre_processor_configs = configs.get("PreProcessor", None)
        if multi_pre_processor_configs != None:
            multi_pre_processor_dispatcher = MultiPreProcessorDispatcher()
            self.multi_pre_processors = multi_pre_processor_dispatcher.obtain_instance(multi_pre_processor_configs)
        else:
            self.multi_pre_processors = {}

    def _init_pre_cascade_processors(self, configs):
        # pre_cascade_processor_configs = configs.get("PreCascadeProcessors", None)
        # if pre_cascade_processor_configs != None:
        #     pre_cascade_processor_dispatcher = PreCascadeProcessorDispatcher()
        #     self.pre_cascade_processors = pre_cascade_processor_dispatcher.obtain_instance(pre_cascade_processor_configs)
        #     return
        self.pre_cascade_processors = []

    def _init_feature_selectors(self, configs):
        multi_feat_selectors_configs = configs.get("FeatureSelector", None)
        if multi_feat_selectors_configs != None:
            mulit_feat_selector_dispatcher = MultiFeatureSelectorDispatcher()
            self.multi_feat_selectors = mulit_feat_selector_dispatcher.obtain_instance(multi_feat_selectors_configs)
        else:
            self.multi_feat_selectors = {}

    def _init_data_and_feature_fusion(self, configs):
        fusion_configs = configs.get("FeatureFusion", None)
        if fusion_configs != None:
            multi_feature_fusion_dispatcher = MultiFeatureFusionDispatcher()
            self.multi_feat_fusions = multi_feature_fusion_dispatcher.obtain_instance(fusion_configs)
        else:
            self.multi_feat_fusions = {}

    def _init_fusion_feature_processors(self, configs):

        fusion_feat_processors_configs = configs.get("FusionFeatureProcessors", None)
        if fusion_feat_processors_configs != None:
            self.fusion_feat_processors = MultiFusionFeatDispatcher.\
                obtain_instance(fusion_feat_processors_configs)
        else:
            self.fusion_feat_processors = {}

    def _init_category_imbalance_processors(self, configs):
        category_imbalance_config = configs.get("CategoryImbalance", None)
        if category_imbalance_config != None:
            category_imbalance_dispatcher = CategoryImbalanceDispatcher()
            self.category_imbalance_processor = category_imbalance_dispatcher.\
                obtain_instance(category_imbalance_config)
        else:
            self.category_imbalance_processor = None

    def _init_cascade_classifier_builder(self, configs):
        builder_configs = configs.get("CascadeClassifier", None)
        if builder_configs != None:
            builder_dispatcher = ClassifierDispatcher()
            self.multi_classifier_builders = builder_dispatcher.obtain_instance(builder_configs)
        else:
            raise "分类器不能为空"

    def _init_cascade_features_processors(self, config):
        feat_processors_configs = config.get("FeatureProcessors", None)
        if feat_processors_configs != None:
            feat_processor_dispatcher = FeaturesProcessorDispatcher()
            self.feature_processors = feat_processor_dispatcher.obtain_instance(feat_processors_configs)
        else:
            self.feature_processors = []

    def _init_cascade_metrics_processors(self, config):
        metrics_processors_configs = config.get("MetricsProcessors", None)
        if metrics_processors_configs != None:
            multi_metrics_processors_dispatcher = MetricProcessorDispatcher()
            self.multi_metrics_processor = multi_metrics_processors_dispatcher.obtain_instance(config=metrics_processors_configs)
        else:
            raise "计算终止指标的方法必须设置"

    def _init_post_cascade_processors(self, configs):
        pre_cascade_processor_configs = configs.get("PreCascadeProcessors", None)
        if pre_cascade_processor_configs != None:
            pre_cascade_processor_dispatcher = PostCascadeProcessorDispatcher()
            self.pre_cascade_processors = pre_cascade_processor_dispatcher.obtain_instance(
                pre_cascade_processor_configs)
            return
        self.pre_cascade_processors = []

    def _init_post_processor(self, config):
        post_processors_configs = config.get("PostProcessors", None)
        if post_processors_configs != None:
            multi_post_processor_dispatcher = MultiPostProcessorDispatcher()
            self.post_processors = multi_post_processor_dispatcher.obtain_instance(post_processors_configs)
        else:
            self.post_processors = {}

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if y_val is None:
            division_ratio = self.config.get("DivisionRatio", 0.8)
            if self.debug:
                print("没有设置验证集， 随机从训练集中进行划分, 划分比例为:", division_ratio)
            X_train, X_val, y_train, y_val = self.train_test_split(X_train, y_train, division_ratio=division_ratio)
        self._fit(X_train, y_train, X_val, y_val)

    def train_test_split(self, Xs, ys, division_ratio):
        sample_size = Xs[0].shape[0]
        indices = np.arange(sample_size)
        np.random.shuffle(indices)

        split_size = int(sample_size * division_ratio)
        train_indices, test_indices = indices[:split_size], indices[split_size:]

        X_train = [X[train_indices, :] for X in Xs]
        y_train = ys[train_indices]

        X_val = [X[test_indices, :] for X in Xs]
        y_val = ys[test_indices]

        return X_train, X_val, y_train, y_val

    def _fit(self, X_train, y_train, X_val, y_val):
        start_time = time.time()
        # 在执行循环前需要先进行一些操作
        data = self.execute_before_fit(X_train, y_train, X_val, y_val)
        # 在循环之前进行一些预处理操作
        data = self.execute_pre_fit_processor(data)
        # 进行循环迭代, 获取
        for layer in range(1, self.max_num_iterations + 1, 1):
            # 在正式进行级联时执行一些操作
            data = self.pre_fit_cascade_data_and_infos(data, layer)
            # 获得的对应的基因筛选
            multi_fselect_ids, mult_fselect_infos = self.execute_feat_selector_processors(data, layer)
            # 对数据集执行特征筛选
            data = self.execute_fit_feature_selection(data, multi_fselect_ids)
            # 保存筛选出的特征
            self.save_multi_fselect_ids(multi_fselect_ids, layer)
            # 执行数据融合处理
            data = self.execute_feature_and_data_fit_fusion(data, layer)
            # 对融合的特征分别进行处理
            data = self.execute_fit_fusion_features_processors(data, layer)
            # 对融合的数据执行类别不平衡的算法, 这个处理涉及到样本数的改变
            data = self.execute_category_imbalance(data, layer)
            # 处理单模态的分类器 (包括机器学习方法和深度学习方法)
            classifier_instances = self.execute_cascade_fit_classifier(data, layer)
            # 获得提取到的特征, 概率特征, 和预测值
            all_finfos = self.obtain_relevant_fit_to_data(data, classifier_instances, layer)
            # 对提取到的特征进行处理
            all_finfos = self.execute_fit_feature_processors(all_finfos, layer)
            # 保存提取到的特征, 概率特征, 和预测值
            data = self.save_relevant_fit_to_data(data, all_finfos, layer)
            # 可能需要对分类器进行一些调整
            classifier_instances, data = self.adjust_cascade_classifier(classifier_instances, data)
            # 保存分类器
            self.save_cascade_classifier(classifier_instances, layer)
            # 计算当前层的终止指标
            metric = self.obtain_current_metric(data, layer)
            # 在进行级联前进行一些数据预处理
            data = self.post_fit_cascade_data_and_infos(data, layer)
            # 级联的层数判断
            if layer == 1:
                count = 0
                best_level, best_metric = layer, metric
                best_level = best_level
                best_metric_processors = self.multi_metrics_processor

            else:
                if metric >= best_metric:
                    count = 0
                    best_level, best_metric = layer, metric
                    best_metric_processors = self.multi_metrics_processor
                else:
                    count = count + 1

            print("第 " + str(layer) + " 层的精度:", metric)

            if count >= self.termination_layer or layer == self.max_num_iterations:
                print("模型的层数 = ", best_level, "最佳的指标 = ", best_metric)
                self.best_level = best_level
                self.best_metric_processors = best_metric_processors
                break

        self.execute_after_fit(data)

        end_time = time.time()
        print("花费的时间:", end_time - start_time)

    def save_relevant_fit_to_data(self, data, all_finfos, layer):
        # 保存当前层提取到的提取到的特征
        data["Finfos"][layer] = all_finfos
        return data

    def execute_before_fit(self, Xs_train, y_train, Xs_val, y_val):

        if self.debug:
            print("==================执行循环前的预处理开始==================")

        # 对运行时中的数据部分进行一些操作
        data = dict()
        size = Xs_train[0].shape[0]
        dims = [X_train.shape[1] for X_train in Xs_train]
        dims_num = len(dims)
        data["Original"] = dict(Xs_train=Xs_train, y_train=y_train, Xs_val=Xs_val, y_val=y_val,
                                Size=size, Dims=dims, DimsNum=dims_num)
        data["Finfos"] = dict()
        data["TrainFinfos"] = dict()
        # 对运行期间的一些信息进行保存
        if self.debug:
            print("原始训练集的大小:"+str(size)+", 模态有"+str(dims_num)+"个, 其维度分别是:" + str(dims))

        return data

    def execute_pre_fit_processor(self, datas):
        if self.debug:
            multi_pre_processor_names = dict()

        for m_name, pre_processors in self.multi_pre_processors.items():
            if self.debug:
                multi_pre_processor_names[m_name] = list()

            for pre_processor in pre_processors:
                if pre_processor.executable():
                    if self.debug:
                        multi_pre_processor_names[m_name].append(pre_processor.obtain_name())

                    datas[m_name] = pre_processor.fit_excecute(datas[m_name])

        if self.debug:
            for m_name, pre_processor_names in multi_pre_processor_names.items():
                print("模态" + str(m_name) + "的相关信息:")
                pre_processor_num = len(pre_processor_names)
                print("循环前的预处理器数量:", pre_processor_num)
                if pre_processor_num > 0:
                    print("循环前的预处理器的名字:", pre_processor_names)
                print("==================执行循环前的预处理结束==================")

        return datas

    def pre_fit_cascade_data_and_infos(self, datas, layer):
        print("==================第" + str(layer) + "层开始执行================")
        # 对级联的每个组件进行调整
        self.change_cascade_components(datas, layer)
        # 对级联的数据进行调整
        datas = self.change_cascade_data(datas, layer)

        return datas

    def change_cascade_components(self, datas, layer):
        self.change_multi_feat_selectors(datas, layer)
        self.change_multi_feat_fusions(datas, layer)
        self.change_fusion_feature_processors(datas, layer)
        self.change_category_imbalance_processor(datas, layer)
        self.change_multi_classifier_builder(datas, layer)
        self.change_feature_processors(datas, layer)
        self.change_multi_metrics_processor(datas, layer)

    def change_multi_feat_selectors(self, datas, layer):
        pass

    def change_multi_feat_fusions(self, datas, layer):
        pass

    def change_fusion_feature_processors(self, datas, layer):
        pass

    def change_feature_processors(self, datas, layer):
        pass

    def change_category_imbalance_processor(self, datas, layer):
        pass

    def change_multi_classifier_builder(self, datas, layer):
        pass

    def change_multi_metrics_processor(self, datas, layer):
        pass

    def change_cascade_data(self, datas, layer):
        return datas

    def execute_feat_selector_processors(self, data, layer):

        if self.debug:
            print("==================特征筛选算法开始执行==================")

        # 获得用于训练特征筛选器的数据集
        original_data = data.get("Original")
        Xs_train, y_train, dims = original_data["Xs_train"], original_data["y_train"], original_data["Dims"]
        multi_fselect_ids, mult_fselect_infos = dict(), dict()
        # 执行特征筛选
        fselect_ids = None
        for m_name, feat_selectors in self.multi_feat_selectors.items():

            # 获得对应模态的数据
            fselect_infos = dict(X_train=Xs_train[m_name], y_train=y_train, Dim=dims[m_name], Names=[])
            # 对每个模态依次执行对应的特征筛选
            for feat_selector in feat_selectors:
                if feat_selector.executable(layer):
                    fselect_ids, fselect_infos = feat_selector.fit_excecute(fselect_ids, fselect_infos, layer)

            multi_fselect_ids[m_name] = fselect_ids
            mult_fselect_infos[m_name] = fselect_infos

        if self.debug:
            for m_name, fselect_infos in mult_fselect_infos.items():

                print("模态" + str(m_name) + "的相关信息:")
                feat_selector_num = fselect_infos.get("SelectedDim", -1)

                if feat_selector_num > 0:
                    fselect_names = fselect_infos["Names"]
                    for fselect_name in fselect_names:
                        print("使用的特征筛选器的名字分别是", fselect_name)
                        if fselect_name == 'EnsembleSelector':
                            self.print_ensemble_selector(fselect_infos['EnsembleSelector'])
                        if fselect_name == 'RecallAttribute':
                            self.print_recall_attribute(fselect_infos['RecallAttribute'])

                print("最终获得的筛选特征数量是: " + str(len(multi_fselect_ids[m_name])))

        return multi_fselect_ids, mult_fselect_infos

    def print_recall_attribute(self, fselect_infos):
        print("召回的数量:", fselect_infos["Num"])
        print("召回率:", fselect_infos["Ratio"])

    def print_ensemble_selector(self, fselect_infos):
        print("EnsembleSelector使用的基分类器分别是:", fselect_infos["BaseSelectorNames"])
        print("EnsembleSelector使用的融合方法是:", fselect_infos["FusionName"])

    def execute_fit_feature_selection(self, data, multi_fselect_ids):

        data["Processed"] = dict(Xs_train=dict(), Xs_val=dict())

        for mname, fselect_ids in multi_fselect_ids.items():
            if fselect_ids != None:
                data["Processed"]["Xs_train"][mname] = copy.deepcopy(data["Original"]["Xs_train"][mname][:, fselect_ids])
                data["Processed"]["Xs_val"][mname] = copy.deepcopy(data["Original"]["Xs_val"][mname][:, fselect_ids])
            else:
                data["Processed"]["Xs_train"][mname] = copy.deepcopy(data["Original"]["Xs_train"][mname])
                data["Processed"]["Xs_val"][mname] = copy.deepcopy(data["Original"]["Xs_val"][mname])

        data["Processed"]["y_train"] = copy.deepcopy(data["Original"]["y_train"])
        data["Processed"]["y_val"] = copy.deepcopy(data["Original"]["y_val"])

        if self.debug:
            print("==================特征筛选算法执行完成==================")

        return data

    def save_multi_fselect_ids(self, multi_fselect_ids, layer):
        self.multi_fselect_ids[layer] = multi_fselect_ids

    def execute_feature_and_data_fit_fusion(self, data, layer):

        assert self.multi_feat_fusions is not None and len(self.multi_feat_fusions) > 0, "当前层没有进行特征融合"

        if self.debug:
            print("==================特征融合开始执行==================")

        for m_name, feat_fusion in self.multi_feat_fusions.items():

            if feat_fusion.executable(layer):

                if self.debug:
                    print("模态" + str(m_name) + "的相关信息:")
                    print("特征融合方法: ", feat_fusion.obtain_name())

                # 获得原始数据 (经过特征筛选的数据)
                processed_train = data["Processed"]["Xs_train"][m_name]
                processed_val = data["Processed"]["Xs_val"][m_name]
                # 获得特征信息
                finfos = data.get("Finfos")
                # 进行特征融合
                fusion_train, fusion_val = feat_fusion.fit_excecute(processed_train, processed_val, finfos, m_name, layer)
                # 将融合的数据封装回去
                data["Processed"]["Xs_train"][m_name] = fusion_train
                data["Processed"]["Xs_val"][m_name] = fusion_val

        if self.debug:
            print("==================特征融合执行完成==================")

        return data

    def execute_fit_fusion_features_processors(self, data, layer):

        if self.debug:
            print("==================融合特征处理器执行开始==================")
            mulit_fusion_features_processor_names = dict()

        # 获得原始数据 (经过特征筛选的数据)
        processed_data = data.get("Processed")
        Xs_train, Xs_val = processed_data.get("Xs_train"), processed_data.get("Xs_val")

        for m_name, fusion_feat_processors in self.fusion_feat_processors.items():
            if self.debug:
                mulit_fusion_features_processor_names[m_name] = []

            for fusion_feat_processor in fusion_feat_processors:
                if self.debug:
                    mulit_fusion_features_processor_names[m_name].append(fusion_feat_processor.obtain_name())

                if fusion_feat_processor.executable(layer):
                    Xs_train[m_name] = fusion_feat_processor.excecute(Xs_train[m_name], layer)
                    Xs_val[m_name] = fusion_feat_processor.excecute(Xs_val[m_name], layer)

        # 将融合的数据封装回去
        data["Processed"]["Xs_train"] = Xs_train
        data["Processed"]["Xs_val"] = Xs_val

        if self.debug:
            for m_name, fusion_features_processor_names in mulit_fusion_features_processor_names.items():
                print("模态" + str(m_name) + "的相关信息:")
                fusion_features_processor_num = len(fusion_features_processor_names)
                print("使用的融合特征处理器的数量为:", fusion_features_processor_num)
                if fusion_features_processor_num > 0:
                    print("使用的特征融合器的名字分别是", fusion_features_processor_names)

            print("==================融合特征处理器执行完成==================")

        return data

    def execute_category_imbalance(self, data, layer):

        if self.category_imbalance_processor is None:
            return data

        if self.debug:
            print("==================类别不平衡器执行开始==================")
            print("类别处理器的名字:", self.category_imbalance_processor.obtain_name())

        if self.category_imbalance_processor.executable(layer):
            # 获得原始数据 (经过特征筛选的数据)
            processed_data = data.get("Processed")
            Xs_train, y_train = processed_data.get("Xs_train"), processed_data.get("y_train")

            X_train_res, y_train_res = self.category_imbalance_processor.fit_excecute(Xs_train, y_train, layer)

            data["Processed"]["Xs_train_res"] = X_train_res
            data["Processed"]["y_train_res"] = y_train_res

        if self.debug:
            print("==================类别不平衡器执行完成==================")

        return data

    def obtain_new_update_builder_configs(self, data, layer):
        new_cfig = dict()
        for classifier_builder in self.multi_classifier_builders:
            builder_name = classifier_builder.obtain_builder_name()
            builder_type = classifier_builder.obtain_builder_type()
            if builder_type == "DL":
                input_size = [X_train.shape[1] for X_train in data["Processed"]["Xs_train"]]
                new_cfig[builder_name] = dict(Model=dict(Parameter=dict(InputSize=input_size)))
        return new_cfig

    def execute_cascade_fit_classifier(self, data, layer):

        if self.multi_classifier_builders is not None and len(self.multi_classifier_builders) != 0:

            if self.debug:
                print("==================分类器执行开始==================")

            classifier_instances = list()

            processed_data = data.get("Processed", None)
            X_train_res = processed_data.get("Xs_train_res", processed_data["Xs_train"])
            y_train_res = processed_data.get("y_train_res", processed_data["y_train"])
            X_val, y_val = processed_data["Xs_val"], processed_data["y_val"]

            for classifier_builder in self.multi_classifier_builders:

                if classifier_builder.fit_executable(layer):
                    m_names = classifier_builder.obtain_modality_name()
                    m_names = " ".join([" ".join(str(m_name)) for m_name in m_names])
                    print("==================分类器对应的模态:" + m_names + "==================")

                    # 获得需要更新的配置参数
                    self.set_new_update_builder_configs(data, classifier_builder, layer)

                    if classifier_builder.fit_executable(layer):
                        # 获得一些额外信息
                        new_cls_data = self.obtain_new_cls_finfos(data, classifier_builder, layer)
                        # 将这些参数分装为一个参数
                        cls_data = dict(X_train=X_train_res, y_train=y_train_res, X_val=X_val, y_val=y_val)
                        cls_data.update(new_cls_data)
                        # 构建训练器
                        cls_instance = classifier_builder.obtain_fit_classifier(cls_data, layer)

                    # 获得训练好的分类器
                    if cls_instance is not None:
                        classifier_instances.append(cls_instance)

            if self.debug:
                totall_num = len(classifier_instances)
                print("训练好的分类器(或特征提取器)总共有" + str(totall_num) + "个")
                print("==================分类器执行完成==================")

            return classifier_instances

        else:
            raise "当前层没有训练任何分类器"

    def set_new_update_builder_configs(self, data, cls_builder, layer):
        builder_type = cls_builder.obtain_builder_type()
        # 对深度学习方法的配置参数进行更新
        if builder_type == "DL":
            # 获得模型参数
            model_cfig = cls_builder.obtain_classifier_cfig()
            # 获得对应的模态
            m_names = cls_builder.obtain_modality_name()
            # 获得更新参数
            Xs_train = data["Processed"]["Xs_train"]
            input_sizes = [X_train.shape[1] for m_name, X_train in Xs_train.items() if m_name in m_names]

            model_cfig["Parameter"].update({"InputSize": input_sizes})

            # 重新设置回去
            cls_builder.set_classifier_cfig(model_cfig)

    def obtain_new_cls_finfos(self, data, cls_builder, layer):

        # 获得当前构建器之前的信息 (该信息是提取的特征, 预测概率值, 预测值等)
        # previous_finfo = self.obtain_new_previous_finfo(data, cls_builder, layer)
        previous_finfo = None
        # 获得当前构造器的训练信息,
        # train_finfo = self.obtain_new_train_finfo(data, cls_builder, layer)
        train_finfo = None
        # 对构建的新信息进行封装
        new_cls_data = dict(TrainFinfo=train_finfo, PreviousFinfo=previous_finfo)

        return new_cls_data

    def obtain_relevant_fit_to_data(self, data, multi_classifier_instances, layer):

        if self.debug:
            print("==================开始获取相关信息(包括提取到的特征, 预测的概率值, 预测值等)==================")

        # 解析当前层所需要的数据
        processed_data = data.get("Processed", None)
        Xs_train, y_train = processed_data["Xs_train"], processed_data["y_train"]
        Xs_val, y_val = processed_data["Xs_val"], processed_data["y_val"]

        all_finfos = []

        for classifier_instance in multi_classifier_instances:
            finfo = self.obtain_current_layer_fit_features(Xs_train, y_train, Xs_val, y_val, classifier_instance, layer)
            all_finfos.append(finfo)

        if self.debug:
            all_features_num = len(all_finfos)
            print("执行特征提取器的过程结束, 最终获得的特征数量为:", all_features_num)
            if all_features_num > 0:
                print("每个特征的属性有", list(all_finfos[0].keys()))
            print("==================获取相关信息结束(包括提取到的特征, 预测的概率值, 预测值等)==================")

        return all_finfos

    def obtain_current_layer_fit_features(self, Xs_train, y_train, Xs_val, y_val, classifier_instance, layer):
        # 真正执行特征提取的地方
        cls_name = classifier_instance.obtain_name()
        m_name = classifier_instance.obtain_modality_name()
        builder_type = classifier_instance.obtain_builder_type()
        classifier_type = classifier_instance.obtain_classifier_type()

        if classifier_instance.can_obtain_features(layer):
            features_train = classifier_instance.obtain_features(Xs_train)
            features_val = classifier_instance.obtain_features(Xs_val)
        else:
            features_train, features_val = None, None

        if classifier_instance.can_obtain_probs(layer):
            probs_train = classifier_instance.predict_probs(Xs_train)
            probs_val = classifier_instance.predict_probs(Xs_val)
        else:
            probs_train, probs_val = None, None

        if classifier_instance.can_obtain_predict(layer):
            predict_train = classifier_instance.predict(Xs_val)
            predict_val = classifier_instance.predict(Xs_val)
        else:
            predict_train, predict_val = None, None

        # 这些信息是必须放的
        finfo = dict(ClassifierName=cls_name, ClassifierType=classifier_type, BuilderType=builder_type,
                      ModalityName=m_name, Feature_train=features_train, Feature_val=features_val, Layer=layer,
                      Predict_train=predict_train, Probs_train=probs_train, Predict_val=predict_val, Probs_val=probs_val)

        return finfo

    def adjust_cascade_classifier(self, classifier_instances, data):
        return classifier_instances, data

    def save_cascade_classifier(self, classifier_instances, layer):
        self.classifier_instances[layer] = classifier_instances

    def execute_fit_feature_processors(self, features, layer):

        if self.debug:
            print("==============开始执行特征处理器===============")
            feature_processors_names = []

        for feat_processor in self.feature_processors:
            if feat_processor.executable(layer):
                if self.debug:
                    feature_processors_names.append(feat_processor.obtain_name())
                features = feat_processor.fit_excecute(features, layer)

        if self.debug:
            feat_processors_num = len(feature_processors_names)
            print("特征处理器的执行数量:", feat_processors_num)
            if feat_processors_num > 0:
                print("特征提取器的执行的名字分别是: ", feature_processors_names)
            print("==============级联后置处理器执行完成===============")

        return features

    def obtain_current_metric(self, data, layer):
        if self.multi_metrics_processor  != None:
            if self.debug:
                print("==============开始执行指标计算器===============")
                print("指标计算器的名字", self.multi_metrics_processor.obtain_name())

            metric = self.multi_metrics_processor.fit_excecute(data, layer)

            if self.debug:
                print("==============指标计算器执行完成===============")

            return metric
        else:
            raise "当前层没有设置指标计算器"

    def post_fit_cascade_data_and_infos(self, data, layer):
        print("==============第" + str(layer) + "层执行结束===============")
        self.all_feature_fusions_processors[layer] = copy.deepcopy(self.multi_feat_fusions)
        self.all_fusion_feature_processors[layer] = copy.deepcopy(self.fusion_feat_processors)
        self.all_feature_processors[layer] = copy.deepcopy(self.feature_processors)
        self.all_metrics_processors[layer] = copy.deepcopy(self.multi_metrics_processor)
        return data

    def execute_after_fit(self, data):
        pass

    def predict(self, X):
        result = np.argmax(self.predict_proba(X), axis=1)
        return result

    def predict_proba(self, X):
        # 在执行循环前需要先进行一些操作
        data = self.execute_before_predict_probs(X)
        # 执行一些数据预处理步骤
        data = self.execute_pre_predict_processor(data)

        for layer in range(1, self.best_level + 1, 1):
            # 在每层进行级联时执行一些相关操作
            data = self.pre_predict_cascade_data_and_infos(data, layer)
            # 获得筛选的特征
            f_select_ids = self.obtain_cascade_f_select_ids(layer)
            # 对数据集执行特征筛选
            data = self.execute_predict_feature_selection(data, f_select_ids)
            # 对融合的特征进行处理
            data = self.execute_feature_and_data_predict_fusion(data, layer)
            # 对融合的特征分别进行处理
            data = self.execute_predict_fusion_features_processors(data, layer)
            # 处理机器学习方法或深度学习方法的模块
            classifier_instances = self.obtain_cascade_predict_classifier_instance(layer)
            # 获得到的提取到的特征
            all_finfos = self.obtain_relevant_to_predict_data(data, classifier_instances, layer)
            # 对提取到的特征进行处理
            all_finfos = self.execute_predict_feature_processors(all_finfos, layer)
            # 保存提取到的特征, 概率特征, 和预测值
            data = self.save_finfos_to_data(data, all_finfos, layer)
            # 在进行级联前进行一些数据预处理
            data = self.post_predict_cascade_data_and_infos(data, layer)

            if layer == self.best_level:
                probs = self.best_metric_processors.predict_execute(data, layer)
                break

        self.execute_after_predict_probs(data)
        return probs

    def execute_before_predict_probs(self, Xs):

        # 对运行时中的数据部分进行一些操作
        data = dict()
        size = Xs[0].shape[0]
        dims = [X.shape[1] for X in Xs]
        dims_num = len(dims)
        data["Original"] = dict(Xs=Xs, Size=size, Dims=dims, DimsNum=dims_num)
        data["Finfos"] = dict()

        if self.debug:
            print("测试集的大小:", size, ", 维度:", dims)

        return data

    def execute_pre_predict_processor(self, datas):
        # 执行预处理代码
        for m_name, pre_processors in self.multi_pre_processors.items():
            for pre_processor in pre_processors:
                if pre_processor.fit_executable():
                    datas[m_name] = pre_processor.fit_excecute(datas[m_name])

        return datas

    def pre_predict_cascade_data_and_infos(self, data, layer):
        print("==================第" + str(layer) + "层开始执行================")
        self.multi_feat_fusions = self.all_feature_fusions_processors[layer]
        self.fusion_feat_processors = self.all_fusion_feature_processors[layer]
        self.feature_processors = self.all_feature_processors[layer]
        self.multi_metrics_processor = self.all_metrics_processors[layer]
        return data

    def obtain_cascade_f_select_ids(self, layer):
        return self.multi_fselect_ids[layer]

    def execute_predict_feature_selection(self, data, multi_fselect_ids):

        data["Processed"] = dict(Xs=dict())

        for mname, fselect_ids in multi_fselect_ids.items():
            if fselect_ids != None:
                data["Processed"]["Xs"][mname] = copy.deepcopy(data["Original"]["Xs"][mname][:, fselect_ids])
            else:
                data["Processed"]["Xs"][mname] = copy.deepcopy(data["Original"]["Xs"][mname])

        return data

    def execute_feature_and_data_predict_fusion(self, data, layer):

        assert self.multi_feat_fusions is not None and len(self.multi_feat_fusions) > 0, "当前层没有进行特征融合"

        for m_name, feat_fusion in self.multi_feat_fusions.items():
            if feat_fusion.executable(layer):
                # 获得原始数据 (经过特征筛选的数据)
                processed_X = data["Processed"]["Xs"][m_name]
                # 获得特征信息
                finfos = data.get("Finfos")
                # 进行特征融合
                fusion_X = feat_fusion.predict_excecute(processed_X, finfos, m_name, layer)
                # 将融合的数据封装回去
                data["Processed"]["Xs"][m_name] = fusion_X

        return data

    def execute_predict_fusion_features_processors(self, data, layer):

        # 获得原始数据 (经过特征筛选的数据)
        Xs = data["Processed"]["Xs"]

        for m_name, fusion_feat_processors in self.fusion_feat_processors.items():

            for fusion_feat_processor in fusion_feat_processors:
                if fusion_feat_processor.executable(layer):
                    Xs[m_name] = fusion_feat_processor.excecute(Xs[m_name], layer)

        # 将融合的数据封装回去
        data["Processed"]["Xs"] = Xs

        return data

    def obtain_cascade_predict_classifier_instance(self, layer):
        return self.classifier_instances[layer]

    def obtain_relevant_to_predict_data(self, data, multi_classifier_instances, layer):
        # 解析当前层所需要的数据
        Xs = data["Processed"]["Xs"]

        all_finfos = []

        for classifier_instance in multi_classifier_instances:
            finfos = self.obtain_current_layer_predict_features(Xs, classifier_instance, layer)
            all_finfos.append(finfos)

        return all_finfos

    def obtain_current_layer_predict_features(self, Xs, classifier_instance, layer):

        # 真正执行特征提取的地方
        cls_name = classifier_instance.obtain_name()
        m_name = classifier_instance.obtain_modality_name()
        builder_type = classifier_instance.obtain_builder_type()
        classifier_type = classifier_instance.obtain_classifier_type()

        if classifier_instance.can_obtain_features(layer):
            features_X = classifier_instance.obtain_features(Xs)
        else:
            features_X = None

        if classifier_instance.can_obtain_probs(layer):
            probs = classifier_instance.predict_probs(Xs)
            predict = classifier_instance.predict(Xs)

        else:
            probs, predict = None, None

        finfo = dict(ClassifierName=cls_name, ClassifierType=classifier_type, BuilderType=builder_type, Layer=layer,
                      ModalityName=m_name, Feature_X=features_X, Predict_val=predict, Probs_val=probs)

        return finfo

    def execute_predict_feature_processors(self, all_finfos, layer):
        for feat_processor in self.feature_processors:
                if feat_processor.executable(layer):
                    for id, feat in enumerate(all_finfos):
                            all_finfos[id] = feat_processor.predict_excecute(feat, layer)
        return all_finfos

    def post_predict_cascade_data_and_infos(self, data, layer):
        print("==============第" + str(layer) + "层执行结束===============")
        return data

    def execute_after_predict_probs(self, data):
        pass

    def set_multi_feat_selectors(self, new_multi_feat_selectors):
        self.multi_feat_selectors = new_multi_feat_selectors

    def obtain_multi_feat_selectors(self):
        return self.multi_feat_selectors

    def set_multi_feat_fusions(self, new_multi_feat_fusions):
        self.multi_feat_fusions = new_multi_feat_fusions

    def obtain_multi_feat_fusions(self):
        return self.multi_feat_fusions

    def set_fusion_feature_processors(self, new_fusion_feat_processors):
        self.fusion_feat_processors = new_fusion_feat_processors

    def obtain_fusion_feature_processors(self):
        return self.fusion_feat_processors

    def set_feature_processors(self, new_feature_processors):
        self.feature_processors = new_feature_processors

    def obtain_feature_processors(self):
        return self.feature_processors

    def set_category_imbalance_processor(self, new_category_imbalance_processor):
        self.category_imbalance_processor = new_category_imbalance_processor

    def obtain_category_imbalance_processor(self):
        return self.category_imbalance_processor

    def set_multi_classifier_builders(self, new_multi_classifier_builders):
        self.multi_classifier_builders = new_multi_classifier_builders

    def obtain_multi_classifier_builders(self):
        return self.multi_classifier_builders

    def obtain_multi_metrics_processor(self, new_multi_metrics_processor):
        self.multi_metrics_processor = new_multi_metrics_processor

    def save_finfos_to_data(self, data, all_finfos, layer):
        data["Finfos"][layer] = all_finfos
        return data
