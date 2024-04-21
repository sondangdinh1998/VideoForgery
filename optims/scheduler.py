import math


def get_lr(cfg, cur_epoch):
    """
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    lr = cosine_annealing(cfg, cur_epoch)

    if cur_epoch < cfg.warmup_epochs:
        lr_start = cfg.warmup_start_lr
        lr_end = cosine_annealing(cfg, cfg.warmup_epochs)
        alpha = (lr_end - lr_start) / cfg.warmup_epochs
        lr = cur_epoch * alpha + lr_start

    return lr


def cosine_annealing(cfg, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return (
        cfg.base_lr
        * (math.cos(math.pi * cur_epoch / cfg.max_epochs) + 1.0)
        * 0.5
    )


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
