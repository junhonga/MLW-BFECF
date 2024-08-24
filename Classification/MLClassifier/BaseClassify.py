import joblib
import numpy as np

from Classification.MLClassifier.RootClassify import BaseClassifierWrapper


def forest_predict_batch_size(clf, X):
    import psutil
    free_memory = psutil.virtual_memory().free
    if free_memory < 2e9:
        free_memory = int(2e9)
    max_mem_size = max(int(free_memory * 0.5), int(8e10))
    mem_size_1 = clf.n_classes_ * clf.n_estimators * 16
    batch_size = (max_mem_size - 1) / mem_size_1 + 1
    if batch_size < 10:
        batch_size = 10
    if batch_size >= X.shape[0]:
        return 0
    return batch_size


def get_ml_base_classifier(name, est_type, config, layer, default=False):

    assert est_type != None, "基分类器的类型必须配置"

    if est_type == "RandomForestClassifier":
        return GCRandomForestClassifier(name, est_type, config, layer)
    elif est_type == "ExtraTreesClassifier":
        return GCExtraTreesClassifier(name, est_type, config, layer)
    elif est_type == "GaussianNBClassifier":
        return GCGaussianNB(name, est_type, config, layer)
    elif est_type == "BernoulliNBClassifier":
        return GCBernoulliNB(name, est_type, config, layer)
    elif est_type == "MultinomialNBClassifier":
        return GCMultinomialNB(name, est_type, config, layer)
    elif est_type == "KNeighborsClassifier":
        return GCKNeighborsClassifier(name, est_type, config, layer)
    elif est_type == "LogisticRegressionClassifier":
        return GCLR(name, est_type, config, layer)
    elif est_type == "GradientBoostingClassifier":
        return GCGradientBoostingClassifier(name, est_type, config, layer)
    elif est_type == "SVCClassifier":
        return GCSVC(name, est_type, config, layer)
    else:
        if default:
            return None
        else:
            raise "暂时不支持" + name + "分类器"

class SKlearnBaseClassifier(BaseClassifierWrapper):

    def __init__(self, name, est_class, config, layer):
        super(SKlearnBaseClassifier, self).__init__(name, est_class, config, layer)

    def _load_model_from_disk(self, cache_path):
        return joblib.load(cache_path)

    def _save_model_to_disk(self, clf, cache_path):
        joblib.dump(clf, cache_path)


class GCExtraTreesClassifier(SKlearnBaseClassifier):
    def __init__(self, name, est_type, config, layer):
        from sklearn.ensemble import ExtraTreesClassifier
        super(GCExtraTreesClassifier, self).__init__(name, ExtraTreesClassifier, config, layer)


    def _default_predict_batch_size(self, clf, X):
        return forest_predict_batch_size(clf, X)


class GCRandomForestClassifier(SKlearnBaseClassifier):
    def __init__(self, name, est_type, config, layer):
        from sklearn.ensemble import RandomForestClassifier
        super(GCRandomForestClassifier, self).__init__(name, RandomForestClassifier, config, layer)

    def _default_predict_batch_size(self, clf, X):
        return forest_predict_batch_size(clf, X)


class GCLR(SKlearnBaseClassifier):
    def __init__(self, name, est_type, config, layer):
        from sklearn.linear_model import LogisticRegression
        super(GCLR, self).__init__(name, LogisticRegression, config, layer)


class GCSGDClassifier(SKlearnBaseClassifier):
    def __init__(self, name, est_type, config, layer):
        from sklearn.linear_model import SGDClassifier
        super(GCSGDClassifier, self).__init__(name, SGDClassifier, config, layer)


class GCXGBClassifier(SKlearnBaseClassifier):
    def __init__(self, name, est_type, config, layer):
        import xgboost as xgb
        super(GCXGBClassifier, self).__init__(name, xgb.XGBClassifier, config, layer)

class GCGaussianNB(SKlearnBaseClassifier):
    def __init__(self, name, est_type, config, layer):
        from sklearn.naive_bayes import GaussianNB
        super(GCGaussianNB, self).__init__(name, GaussianNB, config, layer)

class GCBernoulliNB(SKlearnBaseClassifier):
    def __init__(self, name, est_type, config, layer):
        from sklearn.naive_bayes import BernoulliNB
        super(GCBernoulliNB, self).__init__(name, BernoulliNB, config, layer)

class GCMultinomialNB(SKlearnBaseClassifier):
    def __init__(self, name, est_type, config, layer):
        from sklearn.naive_bayes import MultinomialNB
        super(GCMultinomialNB, self).__init__(name, MultinomialNB, config, layer)

    def _fit(self, est, X, y):
        X = X - np.min(X)
        super(GCMultinomialNB, self)._fit(est, X, y)

    def _predict_proba(self, est, X):
        X = X - np.min(X)
        super(GCMultinomialNB, self)._predict_proba(est, X)

class GCKNeighborsClassifier(SKlearnBaseClassifier):
    def __init__(self, name, est_type, config, layer):
        from sklearn.neighbors import KNeighborsClassifier
        super(GCKNeighborsClassifier, self).__init__(name, KNeighborsClassifier, config, layer)

class GCGradientBoostingClassifier(SKlearnBaseClassifier):
    def __init__(self, name, est_type, config, layer):
        from sklearn.ensemble import GradientBoostingClassifier
        super(GCGradientBoostingClassifier, self).__init__(name, GradientBoostingClassifier, config, layer)

class GCSVC(SKlearnBaseClassifier):
    def __init__(self, name, est_type, config, layer):
        from sklearn.svm import SVC
        super(GCSVC, self).__init__(name, SVC, config, layer)