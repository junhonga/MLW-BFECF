import numpy as np

from Common.Template import RootTemplate


class CategoryBalancerTemplate(RootTemplate):
    pass

class CategoryBalancerWrapper(CategoryBalancerTemplate):

    def __init__(self, name, est_class, est_args):
        self.name = name
        self.est_class = est_class
        self.layers = est_args.pop("layer", None)
        self.est_args = est_args.get("Parameter", dict())
        self.est = None

    def executable(self, layer):
        return True

    def fit_excecute(self, Xs_train, y_train, layer=None):
        dims = {m_name : Xs_train[m_name].shape[1] for m_name, X_train in Xs_train.items()}
        Xs_train = np.concatenate(list(Xs_train.values()), axis=1)
        X_train_res, y_train_res = self.fit(Xs_train, y_train)
        new_train_res = {}
        last_dim = 0
        for m_name, dim in dims.items():
            new_train_res[m_name] = X_train_res[:, last_dim:dim+last_dim]
            last_dim = dim
        return new_train_res, y_train_res

    def _init_estimator(self):
        est = self.est_class(**self.est_args)
        return est

    def _fit(self, est, X, y):
        return est.fit_resample(X, y)

    def fit(self, X, y, cache_dir=None):
        est = self._init_estimator()
        return self._fit(est, X, y)

    def predict_execute(self):
        pass