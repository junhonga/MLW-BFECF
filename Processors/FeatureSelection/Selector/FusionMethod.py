from collections import Counter
import itertools as it

def get_fusion_method(name, est_type, config):
    if est_type == "VoteFusionMethod":
        return VoteFusionMethod(name, config)


class FusionMethod():
    def __init__(self, name, config):
        pass

    def fusion(self, ests_infos):
        pass

class FusionMethodWrapper(FusionMethod):
    def __init__(self, name, config):
        self.name = name

    def execute(self, select_ids, select_infos, select_num):
        pass

class VoteFusionMethod(FusionMethodWrapper):
    def execute(self, select_ids, select_infos, select_num):
        selector_num = len(select_ids)
        standard_selector_num = int(selector_num / 2)

        all_est_ids = []
        for est_name, est_id in select_ids.items():
            all_est_ids.extend(est_id)

        counts = Counter(all_est_ids)
        select_ids = [k for k, v in counts.items() if v > standard_selector_num]
        return select_ids, select_infos


