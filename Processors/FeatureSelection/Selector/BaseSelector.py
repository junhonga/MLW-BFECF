import numpy as np

from Processors.FeatureSelection.Selector.SelectorWrapper import SelectorWrapper


def get_base_selector(name, est_type, config):
    if est_type == "GCLasso":
        return GCLasso(name, config)
    elif est_type == "GCFClassif":
        return GCFClassif(name, config)
    else:
        return None


class GCLasso(SelectorWrapper):
    def __init__(self, name, kwargs):
        from sklearn.linear_model import Lasso
        self.coef = kwargs.get("coef", 0.0001) if kwargs != None else 0.0001
        kwargs = {
            "alpha" : 0.0001, "copy_X" : True, "fit_intercept" : True, "max_iter" : 10000,
            "normalize" : True, "positive" : False, "precompute" : False, "random_state" : None,
            "selection" : 'cyclic', "tol" : 0.0001,  "warm_start" : False
        }
        super(GCLasso, self).__init__(name, Lasso, kwargs)

    def _obtain_selected_index(self, X_train, y_train):
        select_idxs = []
        select_infos = {}
        for ind, coef_ in enumerate(self.est.coef_):
            if np.abs(coef_) > self.coef:
                select_idxs.append(ind)
                select_infos[ind] = coef_
        select_infos["Num"] = len(select_idxs)
        select_infos["Name"] = self.name
        return select_idxs, select_infos

    def obtain_all_index(self, X=None):
        select_infos = {"inds": [], "metrics": []}
        for ind, coef_ in enumerate(self.est.coef_):
            select_infos["inds"].append(ind)
            select_infos["metrics"].append(coef_)
        return select_infos


class GCFClassif(SelectorWrapper):

    def __init__(self, name, kwargs):
        from sklearn.feature_selection import SelectKBest, f_classif
        super(GCLasso, self).__init__(name, None, None)
        self.P = kwargs.get("P", 0.5) if kwargs != None else 0.5
        self.model = SelectKBest(f_classif, k=50)

    def _obtain_selected_index(self, X_train, y_train):
        select_infos = {"inds": [], "metrics": []}
        for ind, p_ in enumerate(self.est.pvalues_):
            if p_ < self.P:
                select_infos["inds"].append(ind)
                select_infos["metrics"].append(p_)
        select_infos["Num"] = len(select_infos["inds"])
        return select_infos

    def obtain_all_index(self, X=None):
        select_infos = {"inds": [], "metrics": []}
        for ind, p_ in enumerate(self.est.pvalues_):
            select_infos["inds"].append(ind)
            select_infos["metrics"].append(p_)
        return select_infos


