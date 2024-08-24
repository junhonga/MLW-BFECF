from Processors.FeatureSelection.Selector.BaseSelector import get_base_selector
from Processors.FeatureSelection.Selector.FusionMethod import get_fusion_method
from Processors.FeatureSelection.Selector.SelectorWrapper import SelectorWrapper


def get_ens_selector(name, est_type, config):
    if est_type == "EnsembleSelector":
        return EnsembleSelector(name, config)

class EnsembleSelector(SelectorWrapper):

    def __init__(self, name, config):
        super(EnsembleSelector, self).__init__(name, None, config)
        self.base_selector_config = config.get("BaseSelector", None)
        self.fusion_method_config = config.get("FusionMethod", None)

    def _init_base_selector(self):
        ests = {}
        for est_name, est_arg in self.base_selector_config.items():
            ests[est_name] = self._get_base_selector(est_name, est_arg)
        return ests

    def _fit(self, X_train, y_train, ests):
        for est_name, est in ests.items():
            est.fit(X_train, y_train)

    def _obtain_selected_index(self, X_train, y_train, ests):
        multi_fselect_ids, multi_select_infos, multi_select_nums = dict(), dict(), dict()
        for est_name, est in ests.items():
            fselect_ids, select_infos = est._obtain_selected_index(X_train, y_train)
            multi_fselect_ids[est_name] = fselect_ids
            multi_select_infos[est_name] = select_infos
            multi_select_nums[est_name] = len(fselect_ids)
        return multi_fselect_ids, multi_select_infos, multi_select_nums

    def _init_fusion_method(self, config):
        est_name = config.get("Name", None)
        est_type = config.get("Type", None)
        est = get_fusion_method(est_name, est_type, config)
        if est is None:
            raise " 暂时不支持" + est_type + "融合器"
        return est_name, est_type, est

    def fit_excecute(self, f_select_ids, f_select_infos, layer):

        X_train, y_train = f_select_infos["X_train"], f_select_infos["y_train"]
        current_input_size, current_input_dim = X_train.shape

        if self.select_inds == None or \
                (current_input_dim != self.original_dim and current_input_size != self.original_size) \
                or self.enforcement:
            ests = self._init_base_selector()
            self._fit(X_train, y_train, ests)
            self.ests = ests
            select_ids, select_infos, select_num = self._obtain_selected_index(X_train, y_train, ests)

            f_select_infos["Names"].append("EnsembleSelector")
            f_select_infos["EnsembleSelector"] = dict(BaseSelectorNames=list(self.obtain_ests_name()),
                                                      SelectInfos=select_infos, Num=select_num)

            est_name, est_type, fusion_method = self._init_fusion_method(self.fusion_method_config)
            f_select_ids, f_select_infos = fusion_method.execute(select_ids, f_select_infos, select_num)

            self.f_select_ids, self.f_select_infos = f_select_ids, f_select_infos

            f_select_infos["EnsembleSelector"].update(FusionName=est_name, FusionType=est_type)
        else:
            f_select_ids, selected_infos = self.select_inds, self.select_infos

        f_select_infos["SelectedDim"] = len(f_select_ids)

        return f_select_ids, f_select_infos

    def _get_base_selector(self, name, config):
        est_type = config.get("Type", None)
        est = get_base_selector(name, est_type, config)
        if est is None:
            raise " 暂时不支持" + est_type + "基特征提取器"
        return est

    def _obtain_ests_indexs(self, X=None):
        ests_infos = {}
        for est_name, est in self.ests.items():
            ests_infos[est_name] = est.obtain_selected_index(X)
        return ests_infos

    def obtain_indexs(self, X=None):
        ests_infos = self._obtain_ests_indexs(X)
        return self.fusion_method.fusion(ests_infos)

    def obtain_ests_name(self):
        return self.ests.keys()

    def obtain_ests_instances(self):
        return self.ests.values()


