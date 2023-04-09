import torch.nn as nn
import json
import re
import os
import shutil

from datetime import datetime

CONFIG_RECORD_DIR = "./config_record"

# ----------------------- utils -----------------------


def save_config():
    if not os.path.exists(CONFIG_RECORD_DIR):
        os.makedirs(CONFIG_RECORD_DIR)

    local_time = re.sub(r"\..*$", "s", str(datetime.today()))
    local_time = re.sub(r":", "h", local_time, 1)
    local_time = re.sub(r":", "m", local_time, 1)
    file_path = os.path.join(
        os.path.normpath(CONFIG_RECORD_DIR),
        f"epoch {ModelAndTrainingConfig.start_epoch} " + local_time + ".py"
    )
    shutil.copy("config.py", file_path)
    # with open(file_path, "w+") as j_io:
    #     json.dump(dict, j_io, indent=4)
    print(f"config saved at {file_path}")

# ----------------------- config -----------------------


class ModelAndTrainingConfig:
    defect_labels = ["crack"]
    labels = ["normal"] + defect_labels
    image_shape = (3, 224, 224)
    noise_dim = (3, 224, 224)
    # default filter depth for the generator down- and up-sampling blocks
    init_channels = 64
    # number of covolution block in discriminator
    n_dis = 6
    # number of domains for image translation, crack, water_seepage, normal
    c_dim = len(defect_labels) + 1
    # number of res_block in bottleneck of image-to-image module
    n_res = 6

    dataset_dir = "./dataset_with_deepcrack"
    result_img_dir = "results"
    checkpoints_dir = "checkpoints"
    batch_size = 1  # change n_batch_to_save_checkpoint as well
    n_batch_to_sample_img = 10
    n_batch_to_save_checkpoint = 250
    save_sample_image = True

    # remember to change start_epoch if model_path is changed
    disc_model_path = None
    gen_model_path = None
    start_epoch = 1

    # from https://github.com/fastai/fastbook/issues/85,
    """
    See the forum for how to use fastai on Windows.
    In particular you always need to set ***num_workers=0*** when creating a DataLoaders
    because Pytorch multiprocessing does not work on Windows.
    """
    dataset_num_workers = 0

    # visualize progress:
    bar_format = "{desc}: {percentage:.1f}%|{bar:15}| {n}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]"

    dataset_drop_last = True
    epochs = 1
    lr = 1e-4
    betas = (0, 0.9)


class DiscLambdas:
    d_wgan_gp_loss = 1.0
    cls_loss_on_real = 5


class GenLambdas:
    g_spatial_cat_similarity_loss = 3

    g_cycle_loss = 10
    g_mask_cycle_loss = 10

    g_mask_vanishing_loss = 3
    g_mask_spatial_constraint_loss = 5

    g_cls_loss_on_gened = 10
    g_wgan_gp_loss = 1
