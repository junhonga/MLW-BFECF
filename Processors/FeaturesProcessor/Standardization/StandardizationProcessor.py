import numpy as np

from Processors.FeaturesProcessor.Standardization.StandardizationProcessorWrapper import StandardizationProcessor


def get_standardizer(name, est_type, configs):
    est_method = configs.get("Method")
    if est_method == "MinMax":
        return MinMaxFeaturesProcessor(name,  configs)
    elif est_method == "Zscore":
        return ZscoreFeaturesProcessor(name,  configs)
    elif est_method == "Robust":
        return RobustFeaturesProcessor(name, configs)
    elif est_method == "UnitVector":
        return UnitVectorFeaturesProcessor(name, configs)
    elif est_method == "DecimalScale":
        return DecimalScaleFeaturesProcessor(name, configs)
    else:
        raise ""


class MinMaxFeaturesProcessor(StandardizationProcessor):

    def excecute_feature_processor(self, feature):
        feature_min = feature.min()
        feature_max = feature.max()
        feature = (feature - feature_min) / (feature_max - feature_min)
        return feature

    def modify_features(self, feature, fmin, fmax):
        gap = fmax - fmin
        if (gap - 0.0) <= (10 ** -5):
            feature = (feature - fmin)
        else:
            feature = (feature - fmin) / (fmax - fmin)
        return feature

class ZscoreFeaturesProcessor(StandardizationProcessor):

    def excecute_feature_processor(self, feature):
        mean = np.mean(feature)
        std_dev = np.std(feature)

        feature = (feature - mean) / std_dev
        return feature

    def modify_features(self, feature, mean, std_dev):
        if (std_dev - 0.0) <= (10 ** -5):
            feature = feature - mean
        else:
            feature = (feature - mean) / std_dev
        return feature

class RobustFeaturesProcessor(StandardizationProcessor):

    def excecute_feature_processor(self, feature):

        median = np.median(feature)
        q1 = np.percentile(feature, 25)
        q3 = np.percentile(feature, 75)

        feature = self.modify_features(feature, median, q1, q3)
        return feature

    def modify_features(self, feature, median, q1, q3):
        gap = q3 - q1
        if (gap - 0.0) <= (10 ** -5):
            feature = feature - median
        else:
            feature = (feature - median) / gap
        return feature

class UnitVectorFeaturesProcessor(StandardizationProcessor):
    def excecute_feature_processor(self, feature):

        norm = np.linalg.norm(feature)

        feature = self.modify_features(feature, norm)
        return feature

    def modify_features(self, feature, norm):

        if (norm - 0.0) <= (10 ** -5):
            feature = feature
        else:
            feature = feature / norm
        return feature

class DecimalScaleFeaturesProcessor(StandardizationProcessor):
    def excecute_feature_processor(self, feature):

        max_abs = np.max(np.abs(feature))

        feature = self.modify_features(feature, max_abs)
        return feature

    def modify_features(self, feature, max_abs):

        if (max_abs - 0.0) <= (10 ** -5):
            feature = feature
        else:
            feature = feature / max_abs
        return feature