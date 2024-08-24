from Processors.CategoryImbalanceProcessor.FusionFeaturesProcessor import CategoryBalancerWrapper

def get_category_balancer(name, est_type, config):
    if est_type == "SMOTE":
        return GCSMOTE(name, config)
    if est_type == "RandomOverSampler":
        return  GCRandomOverSampler(name, config)
    if est_type == "ADASYN":
        return  GCADASYN(name, config)
    if est_type == "RandomUnderSampler":
        return  GCRandomUnderSampler(name, config)
    if est_type == "NearMiss":
        return  GCNearMiss(name, config)

class GCSMOTE(CategoryBalancerWrapper):
    def __init__(self, name, config):
        from imblearn.over_sampling import SMOTE
        super(GCSMOTE, self).__init__(name, SMOTE, config)

class GCRandomOverSampler(CategoryBalancerWrapper):
    def __init__(self, name, config):
        from imblearn.over_sampling import RandomOverSampler
        super(GCRandomOverSampler, self).__init__(name, RandomOverSampler, config)

class GCADASYN(CategoryBalancerWrapper):
    def __init__(self, name, config):
        from imblearn.over_sampling import ADASYN
        super(GCADASYN, self).__init__(name, ADASYN, config)

class GCRandomUnderSampler(CategoryBalancerWrapper):
    def __init__(self, name, config):
        from imblearn.under_sampling import RandomUnderSampler
        super(GCRandomUnderSampler, self).__init__(name, RandomUnderSampler, config)

class GCNearMiss(CategoryBalancerWrapper):
    def __init__(self, name, config):
        from imblearn.under_sampling import NearMiss
        super(GCNearMiss, self).__init__(name, NearMiss, config)