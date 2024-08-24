from Processors.CategoryImbalanceProcessor.CategoryBalancer import get_category_balancer
from Processors.CategoryImbalanceProcessor.FusionFeaturesProcessor import CategoryBalancerTemplate

from Processors.FeatureSelection.Selector.BaseSelector import get_base_selector
from Processors.FeatureSelection.Selector.EnsembleSelector import get_ens_selector
from Processors.FeatureSelection.Selector.RecallAttribute import get_attribute_recall_method
from Processors.FeatureSelection.Selector.SelectorWrapper import FeatureSelectorTemplate

from Processors.FeaturesFusion.ConcatenationFusion import get_feature_concatenation_method, FeaturesFusionTemplate
from Processors.FeaturesProcessor.Standardization.StandardizationProcessor import get_standardizer
from Processors.FeaturesProcessor.Standardization.StandardizationProcessorWrapper import FeaturesProcessorTemplate

from Processors.FusionFeaturesProcessors.FusionFeaturesWrapper import FusionFeaturesTemplate
from Processors.MetricProcessor.MetricProcessor import get_metric_calculator
from Processors.MetricProcessor.MetricProcessorWrapper import MetricProcessorTemplate
from Processors.PostProcessors.PostProcessorsWrapper import PostProcessorTemplate
from Processors.PreProcessors.PreProcessorsWrapper import PreProcessorsTemplate

from Processors.ProcessorDispatcher.ProcessorsDispatcherWrapper import Dispatcher, ListDispatcher, MultiDispatcher

class PreProcessorDispatcher(Dispatcher):

    def __init__(self):
        super(PreProcessorDispatcher, self).__init__(PreProcessorsTemplate)

    def execute_dispatcher_method(self, name, est_type, config):
        pass

class FeatureFusionDispatcher(Dispatcher):

    def __init__(self):
        super(FeatureFusionDispatcher, self).__init__(FeaturesFusionTemplate)

    def execute_dispatcher_method(self, name, est_type, config):
        return get_feature_concatenation_method(name, est_type, config)

class CategoryImbalanceDispatcher(Dispatcher):

    def __init__(self):
        super(CategoryImbalanceDispatcher, self).__init__(CategoryBalancerTemplate)

    def execute_dispatcher_method(self, name, est_type, config):
        return get_category_balancer(name, est_type, config)

class MetricProcessorDispatcher(Dispatcher):

    def __init__(self):
        super(MetricProcessorDispatcher, self).__init__(MetricProcessorTemplate)

    def execute_dispatcher_method(self, name, est_type, config):
        return get_metric_calculator(name, est_type, config)

class PostProcessorDispatcher(Dispatcher):

    def __init__(self):
        super(PostProcessorDispatcher, self).__init__(PostProcessorTemplate)

    def execute_dispatcher_method(self, name, est_type, config):
        pass

class MultiFusionFeatDispatcher(Dispatcher):

    def __init__(self):
        super(MultiFusionFeatDispatcher, self).__init__(FusionFeaturesTemplate)

    def execute_dispatcher_method(self, name, est_type, config):
        return get_feature_concatenation_method(name, est_type, config)


class PreCascadeProcessorDispatcher(ListDispatcher):
    def __init__(self):
        super(FeatSelectorDispatcher, self).__init__(FeatureSelectorTemplate)

    def execute_dispatcher_method(self, name, est_type, config):
        est = get_base_selector(name, est_type, config) \
              or get_ens_selector(name, est_type, config) \
              or get_attribute_recall_method(name, est_type, config)
        if est is None:
            raise "暂时不支持这种" + est_type + "特征处理器"
        return est

class FeatSelectorDispatcher(ListDispatcher):

    def __init__(self):
        super(FeatSelectorDispatcher, self).__init__(FeatureSelectorTemplate)

    def execute_dispatcher_method(self, name, est_type, config):
        est = get_base_selector(name, est_type, config) \
              or get_ens_selector(name, est_type, config) \
              or get_attribute_recall_method(name, est_type, config)
        if est is None:
            raise "暂时不支持这种" + est_type + "特征处理器"
        return est

class FeaturesProcessorDispatcher(ListDispatcher):

    def __init__(self):
        super(FeaturesProcessorDispatcher, self).__init__(FeaturesProcessorTemplate)

    def execute_dispatcher_method(self, name, est_type, config):
        if est_type == "Standardization":
            return get_standardizer(name, est_type, config)
        else:
            raise "暂时不支持这种特征筛选器"

class MultiPreProcessorDispatcher(MultiDispatcher):

    def __init__(self):
        super(MultiPreProcessorDispatcher, self).__init__(PreProcessorDispatcher)

class MultiFeatureSelectorDispatcher(MultiDispatcher):

    def __init__(self):
        super(MultiFeatureSelectorDispatcher, self).__init__(FeatSelectorDispatcher)

class MultiCategoryImbalanceDispatcher(MultiDispatcher):

    def __init__(self):
        super(MultiCategoryImbalanceDispatcher, self).__init__(CategoryImbalanceDispatcher)

class MultiFeatureFusionDispatcher(MultiDispatcher):

    def __init__(self):
        super(MultiFeatureFusionDispatcher, self).__init__(FeatureFusionDispatcher)

class MultiPostProcessorDispatcher(MultiDispatcher):

    def __init__(self):
        super(MultiPostProcessorDispatcher, self).__init__(PostProcessorDispatcher)
