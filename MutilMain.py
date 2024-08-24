import copy

from sklearn.metrics import auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Model import MultiModalityModel

if __name__ == '__main__':
    import pandas as pd

    class_num = 2
    config = {
        "ClassNum": class_num,  # 注意这个参数一定要改
        "ModalityNum" : 3,

        "TerminationLayer" : 3,

        "FeatureSelector": {
            0 : {
                "EnsembleSelector": {
                    "Name" : "EnsembleSelector",
                    "Type": "EnsembleSelector",
                    "BaseSelector" : {
                        "GCLasso": {
                            "Type": "GCLasso",
                            "Order": 0,
                            "Parameter": {},
                        }
                    },
                    "FusionMethod":{
                        "Name" : "VoteFusionMethod",
                        "Type" : "VoteFusionMethod",
                        "Order" : 0,
                        "Parameter": {},
                    },
                    "Order": 0,
                    "Parameter": {},
                },
                "RecallAttribute": {
                    "Name" : "RecallAttribute",
                    "Type": "RecallAttribute",
                    "Order": 1,
                    "Parameter": {},
                }
            },
            1 : {
                "GCLasso": {
                    "Name": "GCLasso",
                    "Type": "GCLasso",
                    "Order": 0,
                    "Parameter": {},
                },
                "RecallAttribute": {
                    "Name": "RecallAttribute",
                    "Type": "RecallAttribute",
                    "Order": 1,
                    "Parameter": {},
                }
            },
            2 : {
                "GCLasso": {
                    "Name": "GCLasso",
                    "Type": "GCLasso",
                    "Order": 0,
                    "Parameter": {},
                },
                "RecallAttribute": {
                    "Name": "RecallAttribute",
                    "Type": "RecallAttribute",
                    "Order": 1,
                    "Parameter": {},
                }
            }
        },

        "FeatureFusion": {
            0: {
                "Name" : "FeatureConcatenation",
                "Layers": 1,
                "BuilderType": ["DL"],
                "Type": "FeatureConcatenation",
            },
            1: {
                "Name": "FeatureConcatenation",
                "Layers": 1,
                "BuilderType": ["DL"],
                "Type": "FeatureConcatenation",
            },
            2: {
                "Name": "FeatureConcatenation",
                "Layers": 1,
                "BuilderType": ["DL"],
                "Type": "FeatureConcatenation",
            }
        },

        "CategoryImbalance": {
            "Name": "SMOTE",
            "Type": "SMOTE",
            "Method": "SMOTE",
            "Parameter": {},
        },

        "FeatureProcessors": {
            "MinMax": {
                "Type" : "Standardization",
                "Method" : "MinMax",
                "ModalityNames" : [[0], [1], [2]],
                "BuilderType" : ["DL"],
                "FeaturesType": []
            },
        },
        #
        "MetricsProcessors": {
            "Name" : "WeightMetricProcessor",
            "ModalityNames" : [[0], [1], [2]],
            "BuilderType" : ["DL", "ML"],
            "Type": "WeightMetricProcessor",
            "ClassifierMethod": "acc",
        },

        "CascadeClassifier": [
        {
            "BuilderName": "MLClassifierBuilder",
            "BuilderType": "ML",
            "ModalityNames": [[0], [1], [2]],
            # 构建的分类器的配置
            "ClassifierConfig" : [{
                "ClassifierName" : "AdaptiveEnsembleClassifyByNum",
                "ClassifierType": "AdaptiveEnsembleClassifyByNum",
                "AdaptiveMethod": "retained_num",
                "CaluateMetric": "acc",

                "BaseClassifier": [
                    {
                        "ClassifierName": "RandomForestClassifier",
                        "ClassifierType": "RandomForestClassifier",
                        "Parameter": {"n_estimators": 100, "criterion": "gini",
                                      "class_weight": None, "random_state": 0},
                    },
                    {
                        "ClassifierName": "ExtraTreesClassifier",
                        "ClassifierType": "ExtraTreesClassifier",
                        "LayerScopes": None,
                        "Parameter": {"n_estimators": 100, "criterion": "gini",
                                      "class_weight": None, "random_state": 0},
                    },

                    {
                        "ClassifierName": "RandomForestClassifier",
                        "ClassifierType": "RandomForestClassifier",
                        "Parameter": {"n_estimators": 100, "criterion": "gini",
                                      "class_weight": None, "random_state": 0},
                    },
                    {
                        "ClassifierName": "ExtraTreesClassifier",
                        "ClassifierType": "ExtraTreesClassifier",
                        "LayerScopes": None,
                        "Parameter": {"n_estimators": 100, "criterion": "gini",
                                      "class_weight": None, "random_state": 0},
                    },
                    # {
                    #     "ClassifierName": "GaussianNBClassifier",
                    #     "ClassifierType": "GaussianNBClassifier",
                    #     "LayerScopes": None,
                    #     "Parameter": {},
                    # },
                    # {
                    #     "ClassifierName": "BernoulliNBClassifier",
                    #     "ClassifierType": "BernoulliNBClassifier",
                    #     "LayerScopes": None,
                    #     "Parameter": {}
                    # },
                    # {
                    #     "ClassifierName": "KNeighborsClassifier_1",
                    #     "ClassifierType": "KNeighborsClassifier",
                    #     "LayerScopes": None,
                    #     "Parameter": {"n_neighbors": 2}
                    # },
                    # {
                    #     "ClassifierName": "KNeighborsClassifier_2",
                    #     "ClassifierType": "KNeighborsClassifier",
                    #     "LayerScopes": None,
                    #     "Parameter": {"n_neighbors": 3}
                    # },
                    # {
                    #     "ClassifierName": "KNeighborsClassifier_3",
                    #     "ClassifierType": "KNeighborsClassifier",
                    #     "LayerScopes": None,
                    #     "Parameter": {"n_neighbors": 5}
                    # },
                    # {
                    #     "ClassifierName": "GradientBoostingClassifier",
                    #     "ClassifierType": "GradientBoostingClassifier",
                    #     "LayerScopes": None,
                    #     "Parameter": {}
                    # },
                    # {
                    #     "ClassifierName": "SVCClassifier_1",
                    #     "ClassifierType": "SVCClassifier",
                    #     "LayerScopes": None,
                    #     "Parameter": {"kernel": "linear", "probability": True}
                    # },
                    # {
                    #     "ClassifierName": "SVCClassifier_2",
                    #     "ClassifierType": "SVCClassifier",
                    #     "LayerScopes": None,
                    #     "Parameter": {"kernel": "rbf", "probability": True}
                    # },
                    # {
                    #     "ClassifierName": "SVCClassifier_3",
                    #     "ClassifierType": "SVCClassifier",
                    #     "LayerScopes": None,
                    #     "Parameter": {"kernel": "sigmoid", "probability": True}
                    # },
                    # {
                    #     "ClassifierName": "LogisticRegressionClassifier_1",
                    #     "ClassifierType": "LogisticRegressionClassifier",
                    #     "LayerScopes": None,
                    #     "Parameter": {"penalty": 'l2'}
                    # },
                    # {
                    #     "ClassifierName": "LogisticRegressionClassifier_2",
                    #     "ClassifierType": "LogisticRegressionClassifier",
                    #     "LayerScopes": None,
                    #     "Parameter": {"C": 1, "penalty": 'l1', "solver": 'liblinear'}
                    # },
                    # {
                    #     "ClassifierName": "LogisticRegressionClassifier_1",
                    #     "ClassifierType": "LogisticRegressionClassifier",
                    #     "LayerScopes": None,
                    #     "Parameter": {"penalty": 'none'}
                    # }
                ],
            }]
        },
        {
            "BuilderName": "DLClassifierBuilder",
            "BuilderType": "DL",
            "ModalityNames": [[0], [1], [2]],
            "Trainer": {
                "name": "Trainer2",
                "Parameter": {},
                "MaxEpoch": 50
            },
            "Model": [{
                "name": "BNN",
                "ModelType": "BNN",
                "Parameter": {"ClassNum": 2, "HiddenParameter": 32}
            }],
            "LossFun": {
                "name": "CrossEntropyLoss",
                "Parameter": {}
            },
            "Optimizer": {
                "name": "Adam",
                "Parameter": {"lr": 0.001},
            }
        },
        {
            "BuilderName": "DLClassifierBuilder",
            "BuilderType": "DL",
            "ModalityNames": [[0, 1], [1, 2], [0, 1]],
            "Trainer": {
                "name": "Trainer2",
                "MaxEpoch": 50
            },
            "Model": [{
                "name": "TwoBNN",
                "ModelType": "TwoBNN",
                "Parameter": {"ClassNum": class_num, "HiddenParameter": 32}
            }],
            "LossFun": {
                "name": "CrossEntropyLoss",
                "Parameter": {}
            },
            "Optimizer": {
                "name": "Adam",
                "Parameter": {"lr": 0.001},
            }
        }],
    }



    def read_data():
        path = r"C:\Users\13241\Downloads\Cancer_prognosis_classification--master" \
               r"\Cancer_prognosis_classification--master\Data"
        import os

        mRNA = pd.read_csv(os.path.join(path, "KIRP_mrna.txt"), sep='\t', )
        # RNA = RNA.set_index("Unnamed: 0")
        # RNA_len = RNA.shape[1]
        methy = pd.read_csv(os.path.join(path, "KIRP_methy.txt"), sep='\t', )
        # CNA = CNA.set_index("Unnamed: 0")
        # CNA_len = RNA_len + CNA.shape[1]
        mirna = pd.read_csv(os.path.join(path, "KIRP_mirna.txt"), sep='\t', )
        # methylation = methylation.set_index("Unnamed: 0")
        # methylation_len = CNA_len + methylation.shape[1]

        target = pd.read_csv(os.path.join(path, "KIRP_label.txt"), sep='\t', )

        print(methy.values[:200].shape)

        # X_train = [methylation.values]
        # X_train = [CNA.values]
        # X_train = [methylation.values]

        X_train = [methy.values[:200], mirna.values[:200], mRNA.values[:200]]
        X_test = [methy.values[200:], mirna.values[200:], mRNA.values[200:]]
        # X_train = [mRNA.values]
        # X_train = [mirna.values]

        # X_test = None

        y_train = target.values[:, 0][:200]
        y_test = target.values[:, 0][200:]

        return X_train, X_test, y_train, y_test


    X_train, X_test, y_train, y_test = read_data()

    model = MultiModalityModel(config)

    model.fit(X_train, y_train, X_test, y_test)

    y_predict = model.predict(X_train)
    micro = accuracy_score(y_predict, y_train)
    print("accuracy_score:", micro)


