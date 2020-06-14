import importlib
import numpy as np
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from classification2d.dataset.dataset import ClassificationDataset
from classification2d.dataset.sampler import EpochConcateSampler
from classification2d.loss.focal_loss import FocalLoss
from classification2d.utils.file_io import load_config, setup_logger
from classification2d.utils.model_io import load_checkpoint, save_checkpoint
from classification2d.network.module.weight_init import kaiming_weight_init


def train(train_config_file):
    """ Medical image segmentation training engine
    :param train_config_file: the input configuration file
    :return: None
    """
    assert os.path.isfile(train_config_file), 'Config not found: {}'.format(train_config_file)

    # load config file
    train_cfg = load_config(train_config_file)

    # clean the existing folder if training from scratch
    model_folder = os.path.join(train_cfg.general.model_save_dir)
    if os.path.isdir(model_folder):
        if train_cfg.general.resume_epoch < 0:
            shutil.rmtree(model_folder)
            os.makedirs(model_folder)
    else:
        os.makedirs(model_folder)

    # copy training and inference config files to the model folder
    shutil.copy(train_config_file, os.path.join(model_folder, 'train_config.py'))
    infer_config_file = os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'infer_config.py'))
    shutil.copy(infer_config_file, os.path.join(train_cfg.general.save_dir, 'infer_config.py'))

    # enable logging
    log_file = os.path.join(model_folder, 'train_log.txt')
    logger = setup_logger(log_file, 'cls2d')

    # control randomness during training
    np.random.seed(train_cfg.general.seed)
    torch.manual_seed(train_cfg.general.seed)
    if train_cfg.general.num_gpus > 0:
        torch.cuda.manual_seed(train_cfg.general.seed)

    # load training dataset
    train_dataset = ClassificationDataset(
        mode='train',
        data_folder=train_cfg.general.train_data_folder,
        label_file=train_cfg.general.train_label_file,
        transforms=train_cfg.dataset.train_transforms
    )
    train_data_loader = DataLoader(
        train_dataset,
        sampler=EpochConcateSampler(train_dataset, train_cfg.train.epochs),
        batch_size=train_cfg.train.batchsize,
        num_workers=train_cfg.train.num_threads,
        pin_memory=True
    )

    # load validation dataset
    val_dataset = ClassificationDataset(
        mode='val',
        data_folder=train_cfg.general.val_data_folder,
        label_file=train_cfg.general.val_label_file,
        transforms=train_cfg.dataset.val_transforms
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    model_module = importlib.import_module('classification2d.network.' + train_cfg.net.name)
    model = model_module.get_detection_model(train_cfg.dataset.num_classes, train_cfg.net.pre_trained)
    kaiming_weight_init(model)

    if train_cfg.general.num_gpus > 0:
        net = nn.parallel.DataParallel(model, device_ids=list(range(train_cfg.general.num_gpus)))
        net = net.cuda()

    # training optimizer
    opt = optim.Adam(model.parameters(), lr=train_cfg.train.lr, betas=train_cfg.train.betas)

    if train_cfg.loss.name == 'Focal':
        # reuse focal loss if exists
        loss_func = FocalLoss(class_num=train_cfg.dataset.num_classes, alpha=train_cfg.loss.obj_weight, gamma=train_cfg.loss.focal_gamma,
                              use_gpu=train_cfg.general.num_gpus > 0)
    else:
        raise ValueError('Unknown loss function')

    # load checkpoint if resume epoch > 0
    if train_cfg.general.resume_epoch >= 0:
        last_save_epoch, batch_start = load_checkpoint(train_cfg.general.resume_epoch, model, opt, model_folder)
    else:
        last_save_epoch, batch_start = 0, 0

    writer = SummaryWriter(os.path.join(model_folder, 'tensorboard'))

    batch_idx = batch_start
    data_iter = iter(train_data_loader)
    # loop over batches
    for i in range(len(train_data_loader)):
        begin_t = time.time()

        crops, masks, frames, filenames = data_iter.next()

        if train_cfg.general.num_gpus > 0:
            crops, masks = crops.cuda(), masks.cuda()

        # clear previous gradients
        opt.zero_grad()

        # network forward and backward
        outputs = net(crops)
        train_loss = loss_func(outputs, masks)
        train_loss.backward()

        # update weights
        opt.step()

        epoch_idx = batch_idx * train_cfg.train.batchsize // len(train_dataset)
        batch_idx += 1
        batch_duration = time.time() - begin_t
        sample_duration = batch_duration * 1.0 / train_cfg.train.batchsize

        # print training loss per batch
        msg = 'epoch: {}, batch: {}, train_loss: {:.4f}, time: {:.4f} s/vol'
        msg = msg.format(epoch_idx, batch_idx, train_loss.item(), sample_duration)
        logger.info(msg)

        writer.add_scalar('Train/Loss', train_loss.item(), batch_idx)

    writer.close()