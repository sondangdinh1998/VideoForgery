import torch
import torch.nn as nn


def seperate_temporal_spatial(module):
    """
    Separate temporal and spatial parameters.
    """
    model_p = {}
    for name, p in module.named_parameters():
        model_p[name] = p

    # go through all attributes of module nn.module (e.g. network or layer)
    # and put batch norms if present
    temporal_p_bn = []
    temporal_p_bn_non = []
    spatial_p_bn = []
    spatial_p_bn_non = []

    for name, m in module.named_modules():
        if isinstance(m, nn.Conv3d):
            kernel_size = list(m.kernel_size)
            assert kernel_size[1] == kernel_size[2]

            if kernel_size[0] == 1 and kernel_size[1] > 1:
                # spatial conv
                if name + ".bias" in model_p.keys():
                    spatial_p_bn_non += [
                        model_p[name + ".weight"],
                        model_p[name + ".bias"],
                    ]
                else:
                    spatial_p_bn_non += [model_p[name + ".weight"]]

            elif kernel_size[0] > 1 and kernel_size[1] == 1:
                # temporal conv
                if name + ".bias" in model_p.keys():
                    temporal_p_bn_non += [
                        model_p[name + ".weight"],
                        model_p[name + ".bias"],
                    ]
                else:
                    temporal_p_bn_non += [model_p[name + ".weight"]]

            else:
                # [1,1,1] conv or [5,7,7] conv
                if name + ".bias" in model_p.keys():
                    spatial_p_bn_non += [
                        model_p[name + ".weight"],
                        model_p[name + ".bias"],
                    ]
                    temporal_p_bn_non += [
                        model_p[name + ".weight"],
                        model_p[name + ".bias"],
                    ]
                else:
                    spatial_p_bn_non += [model_p[name + ".weight"]]
                    temporal_p_bn_non += [model_p[name + ".weight"]]

        elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm3d):
            if "head" in name:
                continue
            elif "norm" in name:
                temporal_p_bn += [model_p[name + ".weight"], model_p[name + ".bias"]]
                spatial_p_bn += [model_p[name + ".weight"], model_p[name + ".bias"]]
            else:
                if name + ".bias" in model_p.keys():
                    temporal_p_bn_non += [
                        model_p[name + ".weight"],
                        model_p[name + ".bias"],
                    ]
                    spatial_p_bn_non += [
                        model_p[name + ".weight"],
                        model_p[name + ".bias"],
                    ]
                else:
                    temporal_p_bn_non += [model_p[name + ".weight"]]
                    spatial_p_bn_non += [model_p[name + ".weight"]]

    for name, p in module.named_parameters():
        if "space_transformer" in name or "space_token" in name:
            spatial_p_bn_non += [model_p[name]]
        elif (
            "temporal_transformer" in name
            or "temporal_token" in name
            or "time_T" in name
        ):
            temporal_p_bn_non += [model_p[name]]
        elif "head" in name or "pos_embedding" in name:
            spatial_p_bn_non += [model_p[name]]
            temporal_p_bn_non += [model_p[name]]

    return temporal_p_bn, temporal_p_bn_non, spatial_p_bn, spatial_p_bn_non


def build_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer
    with momentum for alternating training of Temporal and Spatial kernels.

    Args:
        model: model to perform SGD/ADAM optimization.
        cfg: configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    (
        temporal_p_bn,
        temporal_p_bn_non,
        spatial_p_bn,
        spatial_p_bn_non,
    ) = seperate_temporal_spatial(model)

    # Apply different weight decay to Batchnorm and non-batchnorm parameters.
    # In Caffe2 classification codebase the weight decay for batchnorm is 0.0.
    # Having a different weight decay on batchnorm might cause a performance
    # drop.
    optim_params_temporal = [
        {"params": temporal_p_bn, "weight_decay": 0.0},
        {
            "params": temporal_p_bn_non,
            "weight_decay": cfg.weight_decay,
        },
    ]
    optim_params_spatial = [
        {"params": spatial_p_bn, "weight_decay": 0.0},
        {
            "params": spatial_p_bn_non,
            "weight_decay": cfg.weight_decay,
        },
    ]

    if cfg.method == "sgd":
        return torch.optim.SGD(
            optim_params_temporal,
            lr=cfg.base_lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            dampening=cfg.dampening,
            nesterov=cfg.nesterov,
        ), torch.optim.SGD(
            optim_params_spatial,
            lr=cfg.base_lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            dampening=cfg.dampening,
            nesterov=cfg.nesterov,
        )
    elif cfg.method == "adam":
        return torch.optim.Adam(
            optim_params_temporal,
            lr=cfg.base_lr,
            betas=(0.9, 0.999),
            weight_decay=cfg.weight_decay,
        ), torch.optim.Adam(
            optim_params_spatial,
            lr=cfg.base_lr,
            betas=(0.9, 0.999),
            weight_decay=cfg.weight_decay,
        )
    else:
        raise NotImplementedError(f"Does not support {cfg.method} optimizer")
