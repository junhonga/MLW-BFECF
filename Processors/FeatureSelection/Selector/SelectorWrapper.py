from Common.Template import RootTemplate

class FeatureSelectorTemplate(RootTemplate):
    pass

class SelectorWrapper(FeatureSelectorTemplate):

    def __init__(self, name, est_class, est_args):
        self.name = name
        self.est_class = est_class
        self.est_args = est_args
        self.est = None

        self.select_inds = None
        self.select_infos = None

        self.original_size = None
        self.original_dim = None

        self.enforcement = False

        self.need_info_saved = False

    def _fit(self, X, y):
        self.est.fit(X, y)

    def fit(self, X, y, cache_dir=None):
        self._init_estimator()
        self._fit(X, y)

    def executable(self, layer):
        return True

    def fit_excecute(self, f_select_ids, f_select_infos, layer):

        X_train, y_train = f_select_infos["X_train"], f_select_infos["y_train"]
        current_input_size, current_input_dim = X_train.shape

        if self.select_inds == None or \
          (current_input_dim != self.original_dim and current_input_size != self.original_size) \
           or self.enforcement:
            # 情况一: 第一次执行特征筛选算法
            # 情况二: 说明不是第一次执行特征筛选算法, 且第一层原始数据发生了变化
            # 情况三: 强制执行特征筛选算法 (这个值默认为 False)
            # 这三种情况说明需要重新执行相应的特征筛选算法
            self.fit(X_train, y_train)
            f_select_ids, selected_infos = self._obtain_selected_index(X_train, y_train)
            self.original_size, self.original_dim = current_input_size, current_input_dim

            self.select_inds, self.select_infos = f_select_ids, selected_infos

            name = self.obtain_name()
            f_select_infos[name] = dict(SelectInfos=selected_infos)
            selected_infos["NewTeatures"] = True
        else:
            f_select_ids, selected_infos = self.select_inds, self.select_infos
            selected_infos["NewTeatures"] = False

        f_select_infos["SelectedDim"] = len(f_select_ids)

        return f_select_ids, f_select_infos

    def _init_estimator(self):
        self.est = self.est_class(**self.est_args)

    def _fit(self, X, y):
        self.est.fit(X, y)

    def fit(self, X, y, cache_dir=None):
        self._init_estimator()
        self._fit(X, y)
