from Common.Template import RootTemplate


class FeaturesProcessorTemplate(RootTemplate):
    pass


class StandardizationProcessor(FeaturesProcessorTemplate):

    def __init__(self, name, config):
        self.name = name
        self.est_type = config.get("EstType", [])
        self.builder_type = config.get("BuilderType", [])
        self.feat_type = config.get("FeaturesType", [])

    def executable(self, layer):
        return True

    def fit_excecute(self, finfos, layer):
        for finfo in finfos:
            name = finfo["ClassifierName"]
            if self.determine_processor_finfos(name, finfo, layer):

                features_train = finfo["Feature_train"]
                features_train = self.excecute_feature_processor(features_train)
                finfo["Feature_train"] = features_train

                features_val = finfo.get("Feature_val")
                if features_val is not None:
                    features_val = self.excecute_feature_processor(features_val)
                    finfo["Feature_val"] = features_val

        return finfos

    def predict_excecute(self, finfo, layer):
        name = finfo["ClassifierName"]
        if self.determine_processor_finfos(name, finfo, layer):
            features_X = finfo.get("Feature_X")
            features_X = self.excecute_feature_processor(features_X)
            finfo["Feature_X"] = features_X
        return finfo

    def determine_processor_finfos(self, name, finfo, layer):
        if finfo.get("BuilderType") in self.builder_type:
            return True
        if finfo.get("EstType") in self.feat_type:
            return True
        return False

    def excecute_feature_processor(self, features_X):
        pass
