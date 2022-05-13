config = {}
# set the parameters related to the training and testing set

num_classes = 64

# data_train_opt = {
#     "dataset_name": "MiniImageNet80x80",
#     "nKnovel": 0,
#     "nKbase": num_classes,
#     "n_exemplars": 5,  
#     "n_test_novel": 0,  #一共有多少个图片来自novel class
#     "n_test_base": 64,
#     "batch_size": 1,
#     "epoch_size": 1000,
#     "phase": "train",
# }

data_train_opt = {
    "dataset_name": "MiniImageNet80x80",
    "nKnovel": 0, #5
    "nKbase": 5, #64
    "n_exemplars": 5,  
    "n_test_novel": 0,  
    "n_test_base": 25,
    "batch_size": 1,
    "epoch_size": 1000,
    "phase": "train",
}

#only for test
# data_train_opt = {
#     "dataset_name": "MiniImageNet80x80",
#     "nKnovel": 5,
#     "nKbase": 64, 
#     "n_exemplars": 1,  
#     "n_test_novel": 0,  
#     "n_test_base": 75,
#     "batch_size": 1,
#     "epoch_size": 1000,
#     "phase": "train",
# }
##############


data_test_opt = {
    "dataset_name": "MiniImageNet80x80",
    "nKnovel": 5,
    "nKbase": 0,
    "n_exemplars": 5,
    "n_test_novel": 25,
    "n_test_base": 0,
    "batch_size": 1,
    "epoch_size": 500,
}

config["data_train_opt"] = data_train_opt
config["data_test_opt"] = data_test_opt

config["max_num_epochs"] = 71

LUT_lr = [(20, 0.1), (23, 0.01), (26, 0.001)]

networks = {}
net_optionsF = {"depth": 28, "widen_Factor": 10, "drop_rate": 0.0, "pool": "none"}
net_optim_paramsF = {
    "optim_type": "sgd",
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "LUT_lr": LUT_lr,
}
networks["feature_extractor_1"] = {
    "def_file": "feature_extractors.wide_resnet",
    "pretrained": None,
    "opt": net_optionsF,
    "optim_params": net_optim_paramsF,
}

networks["feature_extractor_2"] = {
    "def_file": "feature_extractors.wide_resnet",
    "pretrained": None,
    "opt": net_optionsF,
    "optim_params": net_optim_paramsF,
}

net_optim_paramsC = {
    "optim_type": "sgd",
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "LUT_lr": LUT_lr,
}
net_optionsC = {
    "num_classes": num_classes,
    "num_features": 640,
    "scale_cls": 10,
    "learn_scale": True,
    "global_pooling": True,
}
networks["classifier"] = {
    # "def_file": "classifiers.cosine_classifier_with_weight_generator",
    "def_file": "classifiers.prototypical_network_head",
    "pretrained": None,
    "opt": net_optionsC,
    "optim_params": net_optim_paramsC,
}

net_optim_paramsC = {
    "optim_type": "sgd",
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "LUT_lr": LUT_lr,
}
net_optionsC = {
    "convnet_type": "wrn_block",
    "convnet_opt": {
        "num_channels_in": 640,
        "num_channels_out": 640,
        "num_layers": 4,
        "stride": 2,
    },
    "classifier_opt": {
        # "classifier_type": "cosine",
        "classifier_type":"linear",
        "num_channels": 640,
        # "scale_cls": 10.0,
        # "learn_scale": True,
        "num_classes": 4,
        "global_pooling": True,
    },
}
networks["classifier_aux"] = {
    "def_file": "classifiers.convnet_plus_classifier",
    "pretrained": None,
    "opt": net_optionsC,
    "optim_params": net_optim_paramsC,
}

config["networks"] = networks

criterions = {"loss": {"ctype": "CrossEntropyLoss", "opt": None}}
config["criterions"] = criterions

config["algorithm_type"] = "selfsupervision.fewshot_selfsupervision_rotation"
config["auxiliary_rotation_task_coef"] = 1.0
config["rotation_invariant_classifier"] = False
config["random_rotation"] = False
