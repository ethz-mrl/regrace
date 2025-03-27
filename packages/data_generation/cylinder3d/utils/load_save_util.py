# -*- coding:utf-8 -*-
# author: Xinge
# @file: load_save_util.py

import torch


def load_checkpoint(model_load_path, model, device=None):
    my_model_dict = model.state_dict()
    if device is not None:
        pre_weight = torch.load(model_load_path,
                                map_location='cuda:' + str(device))
    else:
        pre_weight = torch.load(model_load_path)
    part_load = {}
    match_size = 0
    nomatch_size = 0
    for k in pre_weight.keys():
        value = pre_weight[k]
        if k in my_model_dict and my_model_dict[k].shape == value.shape:
            # print("loading ", k)
            match_size += 1
            part_load[k] = value
        elif k in my_model_dict and my_model_dict[k].shape == value.permute(
                4, 0, 1, 2, 3).shape:
            match_size += 1
            part_load[k] = value.permute(4, 0, 1, 2, 3)
        elif k in my_model_dict and k.split('.')[-2] == 'conv3':
            match_size += 1
            part_load[k] = value[0].unsqueeze(0).permute(4, 0, 1, 2, 3)
        else:
            nomatch_size += 1

    print("matched parameter sets: {}, and no matched: {}".format(
        match_size, nomatch_size))

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)

    return model


def load_checkpoint_1b1(model_load_path, model):
    my_model_dict = model.state_dict()
    pre_weight = torch.load(model_load_path)

    part_load = {}
    match_size = 0
    nomatch_size = 0

    pre_weight_list = [*pre_weight]
    my_model_dict_list = [*my_model_dict]

    for idx in range(len(pre_weight_list)):
        key_ = pre_weight_list[idx]
        key_2 = my_model_dict_list[idx]
        value_ = pre_weight[key_]
        if my_model_dict[key_2].shape == pre_weight[key_].shape:
            # print("loading ", k)
            match_size += 1
            part_load[key_2] = value_
        else:
            print(key_)
            print(key_2)
            nomatch_size += 1

    print("matched parameter sets: {}, and no matched: {}".format(
        match_size, nomatch_size))

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)

    return model
