import json
import os
import os.path as osp
import numpy as np
from datasets import visual_genome
from torch.utils.data import DataLoader


def build_vg_dsets(data_opts, batch_size, image_size, build_canvas=False):
    with open(osp.join(data_opts["root_dir"], data_opts["vocab"]), 'r') as f:
        vocab = json.load(f)
    dset_kwargs = {
        'vocab': vocab,
        'h5_path': osp.join(data_opts["root_dir"], data_opts["trainset"]),
        'image_dir': osp.join(data_opts["root_dir"], data_opts["images"]),
        'image_size': image_size,
        'max_samples': data_opts["num_train_samples"],
        'max_objects': data_opts["max_objects_per_image"],
        'use_orphaned_objects': data_opts["use_orphaned_objects"],
        'use_object_crops': data_opts["use_object_crops"],
        'include_relationships': data_opts["include_relationships"],
        'normalize_method': data_opts.get("normalize_method", "imagenet"),
    }
    if data_opts["use_object_crops"]:
        dset_kwargs['mem_bank_path'] = osp.join(data_opts["root_dir"], data_opts["mem_bank"])
        dset_kwargs['crop_file_csv'] = osp.join(data_opts["root_dir"], data_opts["crops_csv"])
        # dset_kwargs['crop_file_pickle'] = osp.join(data_opts["root_dir"], data_opts["crops_pickle"])
        dset_kwargs['crop_file_pickle'] = data_opts["crops_pickle"]
        dset_kwargs['top10_crop_ids'] = osp.join(data_opts["root_dir"], data_opts["top10_crop_ids"])
        dset_kwargs['build_canvas'] = build_canvas
        dset_kwargs['crop_size'] = data_opts.get("crop_size", None)
        dset_kwargs['retrieve_sampling'] = data_opts.get("retrieve_sampling", "random")
        dset_kwargs['crop_num'] = data_opts.get("crop_num", 1)

    train_dset = visual_genome(**dset_kwargs)
    iter_per_epoch = len(train_dset) // batch_size
    print('There are %d iterations per epoch' % iter_per_epoch)

    # make some modification to dset_kwargs so that it can be used to create val_dset
    dset_kwargs['h5_path'] = osp.join(data_opts["root_dir"], data_opts["valset"])
    dset_kwargs['max_samples'] = data_opts["num_val_samples"]
    # on validation set, we set it to a large number
    dset_kwargs['max_objects'] = 100
    dset_kwargs['top10_crop_ids'] = osp.join(data_opts["root_dir"], data_opts["val_top10_crop_ids"])

    # val_dset = visual_genome(**dset_kwargs)

    return vocab, train_dset


def build_dataset(opts):
    if opts["dataset"] == "visual_genome":
        return build_vg_dsets(opts["data_opts"], opts["batch_size"],
                              opts["image_size"], opts.get("build_canvas", False))
    else:
        raise ValueError("Unrecognized dataset: {}".format(opts["dataset"]))


def build_loaders(opts):
    vocab, train_dset = build_dataset(opts)
    collate_fn = train_dset.collate_fn

    loader_kwargs = {
        'batch_size': opts["batch_size"],
        'num_workers': opts["workers"],
        'shuffle': True,
        'collate_fn': collate_fn,
        "drop_last": True,
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)
    # loader_kwargs['shuffle'] = False
    # val_loader = DataLoader(val_dset, **loader_kwargs)
    return vocab, train_loader


def build_vg_dsets_val(data_opts, batch_size, image_size, build_canvas=False):
    # print(osp.join(data_opts["root_dir"], data_opts["vocab"]), 'ddddd')
    with open(osp.join(data_opts["root_dir"], data_opts["vocab"]), 'r') as f:
        vocab = json.load(f)
    f.close()
    with open(osp.join(data_opts["root_dir"], data_opts["vocab"]), 'r') as f:
        vocab_copy = json.load(f)
    f.close()

    dset_kwargs = {
        'vocab': vocab,
        'h5_path': osp.join(data_opts["root_dir"], data_opts["trainset"]),
        'image_dir': osp.join(data_opts["root_dir"], data_opts["images"]),
        'image_size': image_size,
        'max_samples': data_opts["num_train_samples"],
        'max_objects': data_opts["max_objects_per_image"],
        'use_orphaned_objects': data_opts["use_orphaned_objects"],
        'use_object_crops': data_opts["use_object_crops"],
        'include_relationships': data_opts["include_relationships"],
        'normalize_method': data_opts.get("normalize_method", "imagenet"),
    }
    if data_opts["use_object_crops"]:
        dset_kwargs['mem_bank_path'] = osp.join(data_opts["root_dir"], data_opts["mem_bank"])
        dset_kwargs['crop_file_csv'] = osp.join(data_opts["root_dir"], data_opts["crops_csv"])
        # dset_kwargs['crop_file_pickle'] = osp.join(data_opts["root_dir"], data_opts["crops_pickle"])
        dset_kwargs['crop_file_pickle'] = data_opts["crops_pickle"]
        dset_kwargs['top10_crop_ids'] = osp.join(data_opts["root_dir"], data_opts["top10_crop_ids"])
        dset_kwargs['build_canvas'] = build_canvas
        dset_kwargs['crop_size'] = data_opts.get("crop_size", None)
        dset_kwargs['retrieve_sampling'] = data_opts.get("retrieve_sampling", "random")
        dset_kwargs['crop_num'] = data_opts.get("crop_num", 1)

    train_dset = visual_genome(**dset_kwargs)
    iter_per_epoch = len(train_dset) // batch_size
    print('There are %d iterations per epoch' % iter_per_epoch)

    # make some modification to dset_kwargs so that it can be used to create val_dset
    dset_kwargs['h5_path'] = osp.join(data_opts["root_dir"], data_opts["valset"])
    dset_kwargs['max_samples'] = data_opts["num_val_samples"]
    # on validation set, we set it to a large number
    dset_kwargs['max_objects'] = 100
    dset_kwargs['top10_crop_ids'] = osp.join(data_opts["root_dir"], data_opts["val_top10_crop_ids"])

    dset_kwargs['vocab'] = vocab_copy
    val_dset = visual_genome(**dset_kwargs)

    return vocab, train_dset, val_dset


def build_dataset_val(opts):
    if opts["dataset"] == "visual_genome":
        return build_vg_dsets_val(opts["data_opts"], opts["batch_size"],
                                  opts["image_size"], opts.get("build_canvas", False))
    else:
        raise ValueError("Unrecognized dataset: {}".format(opts["dataset"]))


def build_loaders_eval(opts):
    vocab, train_dset, val_dset = build_dataset_val(opts)
    collate_fn = train_dset.collate_fn

    loader_kwargs = {
        'batch_size': opts["batch_size"],
        'num_workers': opts["workers"],
        'shuffle': True,
        'collate_fn': collate_fn,
        "drop_last": True,
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)
    loader_kwargs['shuffle'] = False
    loader_kwargs['batch_size'] = opts["batch_size"]
    val_loader = DataLoader(val_dset, **loader_kwargs)
    return vocab, train_loader, val_loader
